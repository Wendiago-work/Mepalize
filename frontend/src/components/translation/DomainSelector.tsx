import { Building2 } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useState, useEffect } from 'react'
import { apiService } from '@/services/api'

interface DomainSelectorProps {
  domain: string
  onDomainChange: (domain: string) => void
  disabled?: boolean
}

// Fallback domains in case API is unavailable
const FALLBACK_DOMAINS = [
  { code: 'Game - Music', name: 'Game - Music', description: 'Music-related gaming content and applications' },
  { code: 'Game - Casual', name: 'Game - Casual', description: 'Casual gaming content and applications' },
  { code: 'entertainment', name: 'Entertainment', description: 'General entertainment content' },
]

export function DomainSelector({
  domain,
  onDomainChange,
  disabled = false
}: DomainSelectorProps) {
  const [domains, setDomains] = useState(FALLBACK_DOMAINS)
  const [isLoadingDomains, setIsLoadingDomains] = useState(false)

  useEffect(() => {
    loadDomains()
  }, [])

  const loadDomains = async () => {
    setIsLoadingDomains(true)
    try {
      const domainList = await apiService.getDomains()
      if (domainList && domainList.length > 0) {
        // Convert domain strings to domain objects with descriptions
        const domainObjects = domainList.map(domainCode => ({
          code: domainCode,
          name: domainCode,
          description: getDomainDescription(domainCode)
        }))
        setDomains(domainObjects)
      }
    } catch (error) {
      console.error('Failed to load domains:', error)
      // Keep fallback domains
    } finally {
      setIsLoadingDomains(false)
    }
  }

  const getDomainDescription = (domainCode: string): string => {
    const descriptions: Record<string, string> = {
      'Game - Music': 'Music-related gaming content and applications',
      'Game - Casual': 'Casual gaming content and applications', 
      'entertainment': 'General entertainment content'
    }
    return descriptions[domainCode] || `${domainCode} domain content`
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <Building2 className="h-4 w-4" />
          Domain Selection
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <label className="text-xs font-medium text-muted-foreground mb-2 block">
            Choose Domain
          </label>
          <select
            value={domain}
            onChange={(e) => onDomainChange(e.target.value)}
            disabled={disabled || isLoadingDomains}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {domains.map((dom) => (
              <option key={dom.code} value={dom.code}>
                {dom.name}
              </option>
            ))}
          </select>
          {isLoadingDomains && (
            <div className="text-xs text-muted-foreground mt-1">
              Loading domains...
            </div>
          )}
        </div>
        
        {domain && (
          <div className="text-xs text-muted-foreground bg-muted/50 p-2 rounded">
            <strong>{domains.find(d => d.code === domain)?.name}:</strong>{' '}
            {domains.find(d => d.code === domain)?.description}
          </div>
        )}
        
        <div className="text-xs text-muted-foreground">
          <p>🎯 <strong>Domain Context:</strong> Select the domain to apply relevant style guides and cultural considerations</p>
        </div>
      </CardContent>
    </Card>
  )
}
