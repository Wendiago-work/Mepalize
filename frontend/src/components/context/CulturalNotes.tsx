import { Globe, Info } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface CulturalNote {
  domain: string
  cultural_note: string
  language: string
}

interface CulturalNotesProps {
  notes: CulturalNote[]
  language: string
  domain: string
  isLoading?: boolean
}

export function CulturalNotes({
  notes,
  language,
  domain,
  isLoading = false
}: CulturalNotesProps) {
  const relevantNotes = notes.filter(note => 
    note.language === language && 
    (note.domain === domain || note.domain === 'general')
  )

  if (isLoading) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Globe className="h-4 w-4" />
            Cultural Notes
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent" />
            Loading cultural notes...
          </div>
        </CardContent>
      </Card>
    )
  }

  if (relevantNotes.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Globe className="h-4 w-4" />
            Cultural Notes
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground">
            <Info className="h-4 w-4 inline mr-2" />
            No specific cultural notes found for {language} in {domain} domain.
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <Globe className="h-4 w-4" />
          Cultural Notes
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {relevantNotes.map((note, index) => (
          <div key={index} className="space-y-2">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-xs">
                {note.domain}
              </Badge>
              <span className="text-xs text-muted-foreground">
                {note.language.toUpperCase()}
              </span>
            </div>
            <div className="text-sm text-foreground bg-muted/30 p-3 rounded-md">
              {note.cultural_note}
            </div>
          </div>
        ))}
        
        <div className="text-xs text-muted-foreground">
          <p>🌍 <strong>Cultural Context:</strong> These notes help ensure culturally appropriate translations</p>
        </div>
      </CardContent>
    </Card>
  )
}
