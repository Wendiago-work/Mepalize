import { CheckCircle, Clock, Brain, Globe, Palette, FileText } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface PromptGenerationResult {
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
  execution_time?: number
  pipeline?: string
}

interface TranslationResultProps {
  result: PromptGenerationResult | null
  isLoading?: boolean
  error?: string | null
}

export function TranslationResult({
  result,
  isLoading = false,
  error = null
}: TranslationResultProps) {
  if (isLoading) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <CheckCircle className="h-4 w-4" />
            Prompt Generation
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent" />
            Generating prompt with context...
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className="border-destructive">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2 text-destructive">
            <CheckCircle className="h-4 w-4" />
            Prompt Generation Error
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-destructive bg-destructive/10 p-3 rounded-md">
            {error}
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!result) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <CheckCircle className="h-4 w-4" />
            Prompt Generation
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground">
            Submit text to generate a prompt with context here.
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2">
            <CheckCircle className="h-4 w-4" />
            Prompt Generated
          </CardTitle>
          <div className="flex items-center gap-2">
            {result.execution_time && (
              <Badge variant="outline" className="text-xs">
                <Clock className="h-3 w-3 mr-1" />
                {result.execution_time.toFixed(2)}s
              </Badge>
            )}
            {result.pipeline && (
              <Badge variant="secondary" className="text-xs">
                {result.pipeline}
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Prompt Generation Status */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Status
          </h4>
          <div className="bg-muted/30 p-4 rounded-md">
            <p className="text-sm text-foreground leading-relaxed">
              {result.translation}
            </p>
          </div>
        </div>

        {/* Reasoning */}
        {result.reasoning && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Brain className="h-4 w-4" />
              Generation Details
            </h4>
            <div className="bg-muted/30 p-3 rounded-md">
              <p className="text-sm text-foreground leading-relaxed">
                {result.reasoning}
              </p>
            </div>
          </div>
        )}

        {/* Cultural Notes */}
        {result.cultural_notes && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Globe className="h-4 w-4" />
              Cultural Considerations
            </h4>
            <div className="bg-muted/30 p-3 rounded-md">
              {typeof result.cultural_notes === 'string' ? (
                <p className="text-sm text-foreground leading-relaxed">
                  {result.cultural_notes}
                </p>
              ) : (
                <pre className="text-xs text-foreground overflow-auto max-h-96">
                  {JSON.stringify(result.cultural_notes, null, 2)}
                </pre>
              )}
            </div>
          </div>
        )}

        {/* Style Applied */}
        {result.style_applied && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium flex items-center gap-2">
              <Palette className="h-4 w-4" />
              Style Applied
            </h4>
            <div className="bg-muted/30 p-3 rounded-md">
              <p className="text-sm text-foreground leading-relaxed">
                {result.style_applied}
              </p>
            </div>
          </div>
        )}

        {/* Domain Considerations */}
        {result.domain_considerations && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Domain Considerations</h4>
            <div className="bg-muted/30 p-3 rounded-md">
              <p className="text-sm text-foreground leading-relaxed">
                {result.domain_considerations}
              </p>
            </div>
          </div>
        )}

        {/* RAG Context */}
        {result.rag_context && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Retrieved Context</h4>
            <div className="bg-muted/30 p-3 rounded-md">
              <div className="text-sm text-muted-foreground leading-relaxed space-y-1">
                <p>• Translation Memory: {result.rag_context.translation_memory_count} items</p>
                <p>• Glossaries: {result.rag_context.glossaries_count} items</p>
                <p>• MongoDB Context: {result.rag_context.mongo_context_available ? 'Available' : 'Not available'}</p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
