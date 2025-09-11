import { Eye, Copy, Check } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface PromptPreviewProps {
  prompt: string
  isLoading?: boolean
  error?: string | null
}

export function PromptPreview({
  prompt,
  isLoading = false,
  error = null
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

  if (error) {
    return (
      <Card className="border-destructive">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2 text-destructive">
            <Eye className="h-4 w-4" />
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
          Submit text to generate a prompt with context from Chroma DB and MongoDB.
</div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Eye className="h-5 w-5" />
            Generated Prompt
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
            <p>🔍 <strong>Full Transparency:</strong> This is the complete prompt generated with context from Chroma DB (translation memory & glossaries) and MongoDB (style guides & cultural notes)</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
