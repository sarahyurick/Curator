/**
 * SearchEngine Module
 * Handles Lunr.js integration and search logic with filtering and grouping
 */

class SearchEngine {
    constructor(utils) {
        this.utils = utils;
        this.index = null;
        this.documents = {};
        this.isInitialized = false;
        // v2 schema field names
        this.topics = new Set();        // was categories
        this.tags = new Set();
        this.contentTypes = new Set();  // was documentTypes, from content.type
        this.audiences = new Set();     // was personas, from content.audience
        this.difficulties = new Set();  // from content.difficulty
        this.modalities = new Set();    // from facets.modality
    }
    
    /**
     * Initialize the search engine with documents
     */
    async initialize(documents) {
        try {
            await this.loadLunr();
            this.documents = documents;
            this.collectMetadata();
            this.buildIndex();
            this.isInitialized = true;
        } catch (error) {
            throw error;
        }
    }
    
    /**
     * Collect metadata for filtering using v2 schema with legacy fallbacks
     * v2 schema uses nested fields: content.type, content.audience, facets.modality
     */
    collectMetadata() {
        // Clear existing sets
        this.topics = new Set();
        this.tags = new Set();
        this.contentTypes = new Set();
        this.audiences = new Set();
        this.difficulties = new Set();
        this.modalities = new Set();
        
        Object.values(this.documents).forEach(doc => {
            // Collect topics (v2) or categories (legacy)
            const topics = doc.topics || doc.categories;
            if (topics) {
                if (Array.isArray(topics)) {
                    topics.forEach(topic => this.topics.add(topic));
                } else if (typeof topics === 'string') {
                    topics.split(',').forEach(topic => this.topics.add(topic.trim()));
                }
            }
            
            // Collect tags (same in both schemas)
            if (doc.tags) {
                if (Array.isArray(doc.tags)) {
                    doc.tags.forEach(tag => {
                        // Split space-separated tags and add individually
                        if (typeof tag === 'string' && tag.includes(' ')) {
                            tag.split(' ').forEach(individualTag => {
                                if (individualTag.trim()) {
                                    this.tags.add(individualTag.trim());
                                }
                            });
                        } else if (tag && tag.trim()) {
                            this.tags.add(tag.trim());
                        }
                    });
                } else if (typeof doc.tags === 'string') {
                    // Handle both comma-separated and space-separated tags
                    const allTags = doc.tags.includes(',') 
                        ? doc.tags.split(',')
                        : doc.tags.split(' ');
                    
                    allTags.forEach(tag => {
                        if (tag && tag.trim()) {
                            this.tags.add(tag.trim());
                        }
                    });
                }
            }
            
            // Collect content type: v2 nested (content.type) or legacy flat (content_type)
            const contentType = doc.content?.type || doc.content_type;
            if (contentType) {
                this.contentTypes.add(contentType);
            }
            
            // Collect audience: v2 nested (content.audience) or legacy flat (personas)
            const audience = doc.content?.audience || doc.personas;
            if (audience) {
                if (Array.isArray(audience)) {
                    audience.forEach(a => this.audiences.add(a));
                } else if (typeof audience === 'string') {
                    this.audiences.add(audience);
                }
            }
            
            // Collect difficulty: v2 nested (content.difficulty) or legacy flat (difficulty)
            const difficulty = doc.content?.difficulty || doc.difficulty;
            if (difficulty) {
                this.difficulties.add(difficulty);
            }
            
            // Collect modality: v2 nested (facets.modality) or legacy flat (modality)
            const modality = doc.facets?.modality || doc.modality;
            if (modality) {
                this.modalities.add(modality);
            }
        });
    }
    
    /**
     * Get available filter options using v2 schema field names
     */
    getFilterOptions() {
        return {
            topics: Array.from(this.topics).sort(),
            tags: Array.from(this.tags).sort(),
            contentTypes: Array.from(this.contentTypes).sort(),
            audiences: Array.from(this.audiences).sort(),
            difficulties: Array.from(this.difficulties).sort(),
            modalities: Array.from(this.modalities).sort()
        };
    }
    
    /**
     * Load Lunr.js library if not already loaded
     */
    async loadLunr() {
        if (typeof lunr === 'undefined') {
            await this.utils.loadScript('https://unpkg.com/lunr@2.3.9/lunr.min.js');
        }
    }
    
    /**
     * Build the Lunr search index with v2 schema support
     */
    buildIndex() {
        const documentsArray = Object.values(this.documents);
        const self = this;
        
        this.index = lunr(function() {
            // Define fields with boosting
            this.ref('id');
            this.field('title', { boost: 10 });
            this.field('content', { boost: 5 });
            this.field('summary', { boost: 8 });
            this.field('headings', { boost: 6 });
            this.field('headings_text', { boost: 7 });
            this.field('keywords', { boost: 9 });
            this.field('tags', { boost: 4 });
            this.field('topics', { boost: 3 });         // v2: was categories
            this.field('content_type', { boost: 2 });   // From content.type
            this.field('audience', { boost: 3 });       // v2: was personas, from content.audience
            this.field('difficulty', { boost: 2 });     // From content.difficulty
            this.field('modality', { boost: 2 });       // From facets.modality
            this.field('section_path', { boost: 1 });
            this.field('author', { boost: 1 });
            
            // Add documents to index
            documentsArray.forEach((doc) => {
                try {
                    // Handle v2 schema: content can be object (classification) or string (markdown)
                    // If content is object, use content_text for markdown
                    const markdownContent = typeof doc.content === 'object' 
                        ? (doc.content_text || '').substring(0, 5000)
                        : (doc.content || '').substring(0, 5000);
                    
                    this.add({
                        id: doc.id,
                        title: doc.title || '',
                        content: markdownContent, // Limit content length
                        summary: doc.summary || '',
                        headings: self.extractHeadingsText(doc.headings),
                        headings_text: doc.headings_text || '',
                        keywords: self.arrayToString(doc.keywords),
                        tags: self.arrayToString(doc.tags),
                        // v2 nested fields with legacy fallbacks
                        topics: self.arrayToString(doc.topics || doc.categories),
                        content_type: doc.content?.type || doc.content_type || '',
                        audience: self.arrayToString(doc.content?.audience || doc.personas),
                        difficulty: doc.content?.difficulty || doc.difficulty || '',
                        modality: doc.facets?.modality || doc.modality || '',
                        section_path: self.arrayToString(doc.section_path),
                        author: doc.author || ''
                    });
                } catch (docError) {
                    // Skip documents that fail to index
                }
            }, this);
        });
    }
    
    /**
     * Convert array to string for indexing
     */
    arrayToString(arr) {
        if (Array.isArray(arr)) {
            return arr.join(' ');
        }
        return arr || '';
    }
    
    /**
     * Extract text from headings array
     */
    extractHeadingsText(headings) {
        if (!Array.isArray(headings)) return '';
        return headings.map(h => h.text || '').join(' ');
    }
    
    /**
     * Perform search with query and optional filters
     */
    search(query, filters = {}, maxResults = 20) {
        if (!this.isInitialized || !this.index) {
            return [];
        }
        
        if (!query || query.trim().length < 2) {
            return [];
        }
        
        try {
            // Enhanced search with multiple strategies
            const results = this.performMultiStrategySearch(query);
            
            // Process and enhance results
            const enhancedResults = this.enhanceResults(results, query);
            
            // Apply filters
            const filteredResults = this.applyFilters(enhancedResults, filters);
            
            // Group and rank results
            const groupedResults = this.groupResultsByDocument(filteredResults, query);
            
            return groupedResults.slice(0, maxResults);
                
        } catch (error) {
            return [];
        }
    }
    
    /**
     * Apply filters to search results (v2 schema with legacy fallbacks)
     */
    applyFilters(results, filters) {
        return results.filter(result => {
            // Topic filter (v2) or category filter (legacy)
            if ((filters.topic && filters.topic !== '') || (filters.category && filters.category !== '')) {
                const filterValue = filters.topic || filters.category;
                const docTopics = this.getDocumentTopics(result);
                if (!docTopics.includes(filterValue)) {
                    return false;
                }
            }
            
            // Tag filter
            if (filters.tag && filters.tag !== '') {
                const docTags = this.getDocumentTags(result);
                if (!docTags.includes(filters.tag)) {
                    return false;
                }
            }
            
            // Content type filter (from content.type or content_type)
            if (filters.type && filters.type !== '') {
                const contentType = result.content?.type || result.content_type;
                if (contentType !== filters.type) {
                    return false;
                }
            }
            
            // Audience filter (v2) or persona filter (legacy)
            if ((filters.audience && filters.audience !== '') || (filters.persona && filters.persona !== '')) {
                const filterValue = filters.audience || filters.persona;
                const docAudiences = this.getDocumentAudiences(result);
                if (!docAudiences.includes(filterValue)) {
                    return false;
                }
            }
            
            // Difficulty filter (from content.difficulty or difficulty)
            if (filters.difficulty && filters.difficulty !== '') {
                const difficulty = result.content?.difficulty || result.difficulty;
                if (difficulty !== filters.difficulty) {
                    return false;
                }
            }
            
            // Modality filter (from facets.modality or modality)
            if (filters.modality && filters.modality !== '') {
                const modality = result.facets?.modality || result.modality;
                if (modality !== filters.modality) {
                    return false;
                }
            }
            
            return true;
        });
    }
    
    /**
     * Get topics for a document (v2 schema) with legacy categories fallback
     */
    getDocumentTopics(doc) {
        const topics = [];
        
        // From v2 topics or legacy categories
        const topicsSource = doc.topics || doc.categories;
        if (topicsSource) {
            if (Array.isArray(topicsSource)) {
                topics.push(...topicsSource);
            } else {
                topics.push(...topicsSource.split(',').map(t => t.trim()));
            }
        }
        
        // From section path
        if (doc.section_path && Array.isArray(doc.section_path)) {
            topics.push(...doc.section_path);
        }
        
        // From document ID path
        if (doc.id) {
            const pathParts = doc.id.split('/').filter(part => part && part !== 'index');
            topics.push(...pathParts);
        }
        
        return [...new Set(topics)]; // Remove duplicates
    }
    
    /**
     * Get tags for a document
     */
    getDocumentTags(doc) {
        if (!doc.tags) return [];
        
        if (Array.isArray(doc.tags)) {
            // Handle array of tags that might contain space-separated strings
            const flatTags = [];
            doc.tags.forEach(tag => {
                if (typeof tag === 'string' && tag.includes(' ')) {
                    // Split space-separated tags
                    tag.split(' ').forEach(individualTag => {
                        if (individualTag.trim()) {
                            flatTags.push(individualTag.trim());
                        }
                    });
                } else if (tag && tag.trim()) {
                    flatTags.push(tag.trim());
                }
            });
            return flatTags;
        }
        
        // Handle string tags - check for both comma and space separation
        if (typeof doc.tags === 'string') {
            const allTags = [];
            const tagString = doc.tags.trim();
            
            if (tagString.includes(',')) {
                // Comma-separated tags
                tagString.split(',').forEach(tag => {
                    if (tag.trim()) {
                        allTags.push(tag.trim());
                    }
                });
            } else {
                // Space-separated tags
                tagString.split(' ').forEach(tag => {
                    if (tag.trim()) {
                        allTags.push(tag.trim());
                    }
                });
            }
            
            return allTags;
        }
        
        return [];
    }
    
    
    /**
     * Get audiences for a document (v2 schema) with legacy personas fallback
     */
    getDocumentAudiences(doc) {
        // v2 nested or legacy flat
        const audiences = doc.content?.audience || doc.personas;
        if (!audiences) return [];
        
        if (Array.isArray(audiences)) {
            return audiences;
        }
        
        return [audiences];
    }
    
    /**
     * Perform search with multiple strategies
     */
    performMultiStrategySearch(query) {
        const strategies = [
            // Exact phrase search with wildcards
            `"${query}" ${query}*`,
            // Fuzzy search with wildcards  
            `${query}* ${query}~2`,
            // Individual terms with boost
            query.split(/\s+/).map(term => `${term}*`).join(' '),
            // Fallback: just the query
            query
        ];
        
        let allResults = [];
        const seenIds = new Set();
        
        for (const strategy of strategies) {
            try {
                const results = this.index.search(strategy);
                
                // Add new results (avoid duplicates)
                results.forEach(result => {
                    if (!seenIds.has(result.ref)) {
                        seenIds.add(result.ref);
                        allResults.push({
                            ...result,
                            strategy: strategy
                        });
                    }
                });
                
                // If we have enough good results, stop
                if (allResults.length >= 30) break;
                
            } catch (strategyError) {
                console.warn(`Search strategy failed: ${strategy}`, strategyError);
            }
        }
        
        return allResults;
    }
    
    /**
     * Enhance search results with document data
     */
    enhanceResults(results, query) {
        return results.map(result => {
            const doc = this.documents[result.ref];
            if (!doc) {
                console.warn(`Document not found: ${result.ref}`);
                return null;
            }
            
            return {
                ...doc,
                score: result.score,
                matchedTerms: Object.keys(result.matchData?.metadata || {}),
                matchData: result.matchData,
                strategy: result.strategy
            };
        }).filter(Boolean); // Remove null results
    }
    
    /**
     * Group results by document and find matching sections
     */
    groupResultsByDocument(results, query) {
        const grouped = new Map();
        
        results.forEach(result => {
            const docId = result.id;
            
            if (!grouped.has(docId)) {
                // Find matching sections within this document
                const matchingSections = this.findMatchingSections(result, query);
                
                grouped.set(docId, {
                    ...result,
                    matchingSections,
                    totalMatches: 1,
                    combinedScore: result.score
                });
            } else {
                // Document already exists, combine scores and sections
                const existing = grouped.get(docId);
                const additionalSections = this.findMatchingSections(result, query);
                
                existing.matchingSections = this.mergeSections(existing.matchingSections, additionalSections);
                existing.totalMatches += 1;
                existing.combinedScore = Math.max(existing.combinedScore, result.score);
            }
        });
        
        // Convert map to array and sort by combined score
        return Array.from(grouped.values())
            .sort((a, b) => b.combinedScore - a.combinedScore);
    }
    
    /**
     * Find matching sections within a document
     */
    findMatchingSections(result, query) {
        const matchingSections = [];
        const queryTerms = query.toLowerCase().split(/\s+/);
        
        // Check if title matches
        if (result.title) {
            const titleText = result.title.toLowerCase();
            const hasMatch = queryTerms.some(term => titleText.includes(term));
            
            if (hasMatch) {
                matchingSections.push({
                    type: 'title',
                    text: result.title,
                    level: 1,
                    anchor: ''
                });
            }
        }
        
        // Check headings for matches
        if (result.headings && Array.isArray(result.headings)) {
            result.headings.forEach(heading => {
                const headingText = heading.text?.toLowerCase() || '';
                const hasMatch = queryTerms.some(term => headingText.includes(term));
                
                if (hasMatch) {
                    matchingSections.push({
                        type: 'heading',
                        text: heading.text,
                        level: heading.level || 2,
                        anchor: this.generateAnchor(heading.text)
                    });
                }
            });
        }
        
        // If no specific sections found, add a general content match
        if (matchingSections.length === 0) {
            matchingSections.push({
                type: 'content',
                text: 'Content match',
                level: 0,
                anchor: ''
            });
        }
        
        return matchingSections;
    }
    
    /**
     * Generate anchor link similar to how Sphinx does it
     */
    generateAnchor(headingText) {
        if (!headingText) return '';
        
        return headingText
            .toLowerCase()
            .replace(/[^\w\s-]/g, '')  // Remove special chars
            .replace(/\s+/g, '-')      // Replace spaces with hyphens
            .trim();
    }
    
    /**
     * Merge sections, avoiding duplicates
     */
    mergeSections(existing, additional) {
        const merged = [...existing];
        
        additional.forEach(section => {
            const isDuplicate = existing.some(existingSection => 
                existingSection.text === section.text && 
                existingSection.type === section.type
            );
            
            if (!isDuplicate) {
                merged.push(section);
            }
        });
        
        return merged;
    }
    
    /**
     * Get search statistics
     */
    getStatistics() {
        return {
            documentsIndexed: Object.keys(this.documents).length,
            topicsAvailable: this.topics.size,
            tagsAvailable: this.tags.size,
            contentTypesAvailable: this.contentTypes.size,
            audiencesAvailable: this.audiences.size,
            isInitialized: this.isInitialized
        };
    }
    
    /**
     * Check if the search engine is ready
     */
    isReady() {
        return this.isInitialized && this.index !== null;
    }
}

// Make SearchEngine available globally
window.SearchEngine = SearchEngine; 