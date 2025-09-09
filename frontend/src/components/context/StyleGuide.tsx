import { Palette, Info } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface StyleGuideData {
  _id?: string
  domain: string
  style_guide?: any 
  created_at?: string
  updated_at?: string
}

interface StyleGuideProps {
  styleGuide: StyleGuideData | null
  domain: string
  isLoading?: boolean
}


export function StyleGuide({
  styleGuide,
  domain,
  isLoading = false
}: StyleGuideProps) {
  if (isLoading) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Palette className="h-4 w-4" />
            Style Guide
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent" />
            Loading style guide...
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!styleGuide) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Palette className="h-4 w-4" />
            Style Guide
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground">
            <Info className="h-4 w-4 inline mr-2" />
            No style guide found for {domain} domain.
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <Palette className="h-4 w-4" />
          Style Guide
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="text-xs">
            {styleGuide.domain}
          </Badge>
        </div>
        
        {styleGuide.style_guide && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Style Guide:</h4>
            <pre className="text-xs text-foreground bg-muted/50 p-3 rounded-md overflow-auto max-h-96">
              {JSON.stringify(styleGuide.style_guide, null, 2)}
            </pre>
          </div>
        )}
        
        <div className="text-xs text-muted-foreground">
          <p>🎨 <strong>Style Context:</strong> These guidelines ensure consistent tone and style in translations</p>
        </div>
      </CardContent>
    </Card>
  )
}
