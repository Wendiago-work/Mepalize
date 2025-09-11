const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

export interface ImageAttachment {
  id: string
  file: File
  preview: string
  name: string
  size: number
}

export interface TranslationRequest {
  text: string
  source_language: string
  target_language: string
  domain: string
  context_notes?: string
  attachments?: Array<{
    type: string
    base64_data?: string
    filename?: string
    mime_type?: string
    description?: string
  }>
}

export interface PromptGenerationResponse {
  session_id: string
  message_id: string
  timestamp: string
  translation: string  // Placeholder text indicating prompt was generated
  reasoning?: string
  cultural_notes?: string
  style_applied?: string
  domain_considerations?: string
  full_prompt: string  // The main output - the generated prompt
  rag_context?: {
    translation_memory_count: number
    glossaries_count: number
    mongo_context_available: boolean
  }
  execution_time: number
  pipeline: string
}

export interface CulturalNote {
  domain: string
  cultural_note: any  
  language: string
}

export interface StyleGuide {
  domain: string
  tone: string
  guidelines: string[]
}

export interface DataSummary {
  mongodb: {
    collections: {
      style_guides: number
      cultural_notes: number
    }
    domains: string[]
    languages: string[]
  }
  chroma_db: {
    translation_memory_count: number
    glossaries_count: number
    collections: string[]
  }
  total_knowledge: {
    mongodb_documents: number
    chroma_documents: number
    translation_memory_entries: number
    glossary_entries: number
  }
}

class ApiService {
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_BASE}${endpoint}`
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    })

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`)
    }

    return response.json()
  }


  async generatePrompt(request: TranslationRequest): Promise<PromptGenerationResponse> {
    return this.request<PromptGenerationResponse>('/chat/translate', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  async generatePromptWithImages(
    text: string,
    sourceLanguage: string,
    targetLanguage: string,
    domain: string,
    contextNotes?: string,
    attachments?: ImageAttachment[]
  ): Promise<PromptGenerationResponse> {
    const formData = new FormData()
    
    // Debug: Log what we're sending
    console.log('🔍 Frontend sending:', {
      text: `'${text}'`,
      sourceLanguage,
      targetLanguage,
      domain,
      contextNotes: `'${contextNotes || ''}'`,
      attachmentsCount: attachments?.length || 0
    })
    
    // Add text fields
    formData.append('text', text)
    formData.append('source_language', sourceLanguage)
    formData.append('target_language', targetLanguage)
    formData.append('domain', domain)
    formData.append('context_notes', contextNotes || '')
    
    // Add image attachments as files
    if (attachments && attachments.length > 0) {
      attachments.forEach(attachment => {
        formData.append('attachments', attachment.file, attachment.name)
      })
    }

    const url = `${API_BASE}/chat/translate-multipart`
    console.log('🔍 Frontend calling URL:', url)
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      // Don't set Content-Type header - let browser set it with boundary
    })

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`)
    }

    return response.json()
  }

  async getDataSummary(): Promise<DataSummary> {
    return this.request<DataSummary>('/data/summary')
  }

  async getCulturalNotes(language: string): Promise<CulturalNote[]> {
    const response = await this.request<{
      language: string
      cultural_notes: CulturalNote[]
      count: number
      timestamp: string
    }>(`/context/cultural-notes/${language}`)
    return response.cultural_notes
  }

  async getStyleGuide(domain: string): Promise<StyleGuide | null> {
    const response = await this.request<{
      domain: string
      style_guide: StyleGuide | null
      found: boolean
      timestamp: string
    }>(`/context/style-guide/${domain}`)
    return response.style_guide
  }

  async getDomains(): Promise<string[]> {
    const response = await this.request<{
      domains: string[]
      count: number
      timestamp: string
    }>('/context/domains')
    return response.domains
  }

  async getLanguages(): Promise<string[]> {
    const response = await this.request<{
      languages: string[]
      count: number
      timestamp: string
    }>('/context/languages')
    return response.languages
  }

  async healthCheck(): Promise<{ status: string; message: string }> {
    return this.request<{ status: string; message: string }>('/health')
  }
}

export const apiService = new ApiService()
