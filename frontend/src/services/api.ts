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

  private async convertImageToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        const result = reader.result as string
        // Remove data:image/...;base64, prefix
        const base64 = result.split(',')[1]
        resolve(base64)
      }
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  }

  private async processAttachments(attachments?: ImageAttachment[]): Promise<Array<{
    type: string
    base64_data?: string
    filename?: string
    mime_type?: string
    description?: string
  }>> {
    if (!attachments || attachments.length === 0) return []

    const processedAttachments = await Promise.all(
      attachments.map(async (attachment) => {
        const base64Data = await this.convertImageToBase64(attachment.file)
        return {
          type: 'image',
          base64_data: base64Data,
          filename: attachment.name,
          mime_type: attachment.file.type,
          description: `Image: ${attachment.name} (${Math.round(attachment.size / 1024)}KB)`
        }
      })
    )

    return processedAttachments
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
    const processedAttachments = await this.processAttachments(attachments)
    
    const request: TranslationRequest = {
      text,
      source_language: sourceLanguage,
      target_language: targetLanguage,
      domain,
      context_notes: contextNotes,
      attachments: processedAttachments
    }

    return this.generatePrompt(request)
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
