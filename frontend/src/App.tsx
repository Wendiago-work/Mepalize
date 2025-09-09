import { useState, useEffect } from 'react'
import { Globe, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

// Import components
import { LanguageSelector } from '@/components/translation/LanguageSelector'
import { DomainSelector } from '@/components/translation/DomainSelector'
import { TranslationForm } from '@/components/translation/TranslationForm'
import { TranslationResult } from '@/components/translation/TranslationResult'
import { CulturalNotes } from '@/components/context/CulturalNotes'
import { StyleGuide } from '@/components/context/StyleGuide'
import { PromptPreview } from '@/components/context/PromptPreview'

// Import API service
import { apiService } from '@/services/api'
import type { TranslationResponse, CulturalNote, StyleGuide as StyleGuideType, ImageAttachment } from '@/services/api'

import './App.css'

function App() {
  // Translation state
  const [sourceLanguage, setSourceLanguage] = useState('en')
  const [targetLanguage, setTargetLanguage] = useState('ja')
  const [domain, setDomain] = useState('Game - Music')
  const [isTranslating, setIsTranslating] = useState(false)
  const [translationResult, setTranslationResult] = useState<TranslationResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Context state
  const [culturalNotes, setCulturalNotes] = useState<CulturalNote[]>([])
  const [styleGuide, setStyleGuide] = useState<StyleGuideType | null>(null)
  const [isLoadingContext, setIsLoadingContext] = useState(false)

  // UI state
  const [systemStatus, setSystemStatus] = useState<'healthy' | 'unhealthy' | 'loading'>('loading')

  // Load initial context data
  useEffect(() => {
    loadContextData()
    checkSystemHealth()
  }, [sourceLanguage, targetLanguage, domain])

  const checkSystemHealth = async () => {
    try {
      const health = await apiService.healthCheck()
      setSystemStatus(health.status === 'healthy' ? 'healthy' : 'unhealthy')
    } catch (err) {
      setSystemStatus('unhealthy')
    }
  }

  const loadContextData = async () => {
    setIsLoadingContext(true)
    try {
      // Load cultural notes and style guide
      const [notes, guide] = await Promise.all([
        apiService.getCulturalNotes(targetLanguage).catch(() => []),
        apiService.getStyleGuide(domain).catch(() => null)
      ])
      
      setCulturalNotes(notes)
      setStyleGuide(guide)
    } catch (err) {
      console.error('Failed to load context data:', err)
    } finally {
      setIsLoadingContext(false)
    }
  }

  const handleTranslation = async (text: string, contextNotes?: string, attachments?: ImageAttachment[]) => {
    setIsTranslating(true)
    setError(null)
    setTranslationResult(null)

    try {
      const result = await apiService.translateWithImages(
        text,
        sourceLanguage,
        targetLanguage,
        domain,
        contextNotes,
        attachments
      )
      setTranslationResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Translation failed')
    } finally {
      setIsTranslating(false)
    }
  }

  const clearTranslation = () => {
    setTranslationResult(null)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                <Globe className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <CardTitle className="text-xl">Localized Translator</CardTitle>
                <p className="text-sm text-muted-foreground">AI-powered translation with cultural context</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Badge 
                variant={systemStatus === 'healthy' ? 'default' : 'destructive'}
                className="text-xs"
              >
                {systemStatus === 'loading' ? 'Checking...' : 
                 systemStatus === 'healthy' ? 'System Online' : 'System Offline'}
              </Badge>
              <Button
                variant="outline"
                size="sm"
                onClick={clearTranslation}
              >
                <RotateCcw className="h-4 w-4 mr-2" />
                Clear
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Configuration Panel */}
      <div className="border-b bg-muted/30">
        <div className="container mx-auto px-4 py-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <LanguageSelector
              sourceLanguage={sourceLanguage}
              targetLanguage={targetLanguage}
              onSourceChange={setSourceLanguage}
              onTargetChange={setTargetLanguage}
              disabled={isTranslating}
            />
            <DomainSelector
              domain={domain}
              onDomainChange={setDomain}
              disabled={isTranslating}
            />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Input and Context */}
          <div className="space-y-6">
            <TranslationForm
              onSubmit={handleTranslation}
              isLoading={isTranslating}
              disabled={systemStatus !== 'healthy'}
            />
            
            <CulturalNotes
              notes={culturalNotes}
              language={targetLanguage}
              domain={domain}
              isLoading={isLoadingContext}
            />
            
            <StyleGuide
              styleGuide={styleGuide}
              domain={domain}
              isLoading={isLoadingContext}
            />
          </div>

          {/* Center Column - Results */}
          <div className="space-y-6">
            <TranslationResult
              result={translationResult}
              isLoading={isTranslating}
              error={error}
            />
          </div>

          {/* Right Column - Prompt Preview */}
          <div className="space-y-6">
            <PromptPreview
              prompt={translationResult?.full_prompt || ''}
              isLoading={isTranslating}
            />
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t bg-card">
        <div className="container mx-auto px-4 py-3">
          <div className="text-xs text-muted-foreground text-center">
            <p>Powered by Gemini Pro • Enhanced with RAG • Built with React & FastAPI</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App