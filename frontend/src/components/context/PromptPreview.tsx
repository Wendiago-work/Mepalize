import { Eye, Copy, Check } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface PromptPreviewProps {
  prompt: string
  isLoading?: boolean
}

export function PromptPreview({
  prompt,
  isLoading = false
}: PromptPreviewProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(prompt)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy prompt:', err)
    }
  }

  if (isLoading) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Eye className="h-4 w-4" />
            Final Prompt Preview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent" />
            Generating prompt...
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!prompt) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Eye className="h-4 w-4" />
            Final Prompt Preview
          </CardTitle>
        </CardHeader>
        <CardContent>
        <div className="text-sm text-muted-foreground">
          Submit a translation to see the complete prompt that will be sent to the LLM.
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
            <Eye className="h-4 w-4" />
            Final Prompt Preview
          </CardTitle>
          <Button
            variant="outline"
            size="sm"
            onClick={handleCopy}
            className="h-7 px-2"
          >
            {copied ? (
              <>
                <Check className="h-3 w-3 mr-1" />
                Copied
              </>
            ) : (
              <>
                <Copy className="h-3 w-3 mr-1" />
                Copy
              </>
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs">
              {prompt.length} characters
            </Badge>
            <Badge variant="outline" className="text-xs">
              {prompt.split(' ').length} words
            </Badge>
          </div>
          
          <div className="bg-muted/30 p-4 rounded-md">
            <pre className="text-sm text-foreground whitespace-pre-wrap font-mono leading-relaxed">
              {prompt}
            </pre>
          </div>
          
          <div className="text-xs text-muted-foreground">
            <p>🔍 <strong>Full Transparency:</strong> This is the complete prompt sent to the LLM, including all context, instructions, and RAG data</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
