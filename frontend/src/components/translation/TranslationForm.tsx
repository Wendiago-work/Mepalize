import { useState, useRef, useCallback } from 'react'
import { Send, Loader2, FileText, X, Image as ImageIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

interface ImageAttachment {
  id: string
  file: File
  preview: string
  name: string
  size: number
}

interface TranslationFormProps {
  onSubmit: (text: string, contextNotes?: string, attachments?: ImageAttachment[]) => void
  isLoading: boolean
  disabled?: boolean
}

export function TranslationForm({
  onSubmit,
  isLoading,
  disabled = false
}: TranslationFormProps) {
  const [text, setText] = useState('')
  const [contextNotes, setContextNotes] = useState('')
  const [attachments, setAttachments] = useState<ImageAttachment[]>([])
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = () => {
    if ((!text.trim() && attachments.length === 0) || isLoading) return
    onSubmit(text, contextNotes || undefined, attachments.length > 0 ? attachments : undefined)
    setText('')
    setContextNotes('')
    setAttachments([])
  }


  const handleFileSelect = useCallback((files: FileList | null) => {
    if (!files) return

    Array.from(files).forEach(file => {
      if (file.type.startsWith('image/')) {
        const reader = new FileReader()
        reader.onload = (e) => {
          const preview = e.target?.result as string
          const attachment: ImageAttachment = {
            id: Math.random().toString(36).substr(2, 9),
            file,
            preview,
            name: file.name,
            size: file.size
          }
          setAttachments(prev => [...prev, attachment])
        }
        reader.readAsDataURL(file)
      }
    })
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    handleFileSelect(e.dataTransfer.files)
  }, [handleFileSelect])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const removeAttachment = (id: string) => {
    setAttachments(prev => prev.filter(att => att.id !== id))
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const items = e.clipboardData?.items
    if (!items) return

    for (let i = 0; i < items.length; i++) {
      const item = items[i]
      if (item.type.startsWith('image/')) {
        e.preventDefault()
        const file = item.getAsFile()
        if (file) {
          // Create a mock FileList-like object
          const mockFileList = {
            0: file,
            length: 1,
            item: (index: number) => index === 0 ? file : null,
            [Symbol.iterator]: function* () {
              yield file
            }
          } as FileList
          handleFileSelect(mockFileList)
        }
      }
    }
  }, [handleFileSelect])

  const handleTextareaPaste = useCallback((e: React.ClipboardEvent) => {
    const items = e.clipboardData?.items
    if (!items) return

    for (let i = 0; i < items.length; i++) {
      const item = items[i]
      if (item.type.startsWith('image/')) {
        e.preventDefault()
        const file = item.getAsFile()
        if (file) {
          // Create a mock FileList-like object
          const mockFileList = {
            0: file,
            length: 1,
            item: (index: number) => index === 0 ? file : null,
            [Symbol.iterator]: function* () {
              yield file
            }
          } as FileList
          handleFileSelect(mockFileList)
        }
      }
    }
  }, [handleFileSelect])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Handle Enter key for submission
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <Card
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={`transition-colors ${isDragOver ? 'ring-2 ring-primary ring-offset-2' : ''}`}
    >
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <FileText className="h-4 w-4" />
          Translation Input
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <label className="text-xs font-medium text-muted-foreground mb-2 block">
            Text to Translate
          </label>
          <Textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handleTextareaPaste}
            placeholder="Enter text to translate, paste images, or drag & drop images here..."
            className="resize-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
            rows={4}
            disabled={disabled || isLoading}
          />
        </div>

        {/* Image Upload Section */}
        <div>
          <label className="text-xs font-medium text-muted-foreground mb-2 block">
            Images
          </label>
          <div
            className={`border-2 border-dashed rounded-lg p-4 text-center transition-colors ${
              isDragOver
                ? 'border-primary bg-primary/5'
                : 'border-muted-foreground/25 hover:border-muted-foreground/50'
            } ${disabled || isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
            onClick={() => !disabled && !isLoading && fileInputRef.current?.click()}
            onPaste={handlePaste}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={(e) => handleFileSelect(e.target.files)}
              className="hidden"
              disabled={disabled || isLoading}
            />
            <ImageIcon className="h-6 w-6 mx-auto mb-2 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              Click to upload images or paste from clipboard
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              PNG, JPG, GIF up to 10MB each
            </p>
          </div>
        </div>

        {/* Image Previews */}
        {attachments.length > 0 && (
          <div>
            <label className="text-xs font-medium text-muted-foreground mb-2 block">
              Uploaded Images ({attachments.length})
            </label>
            <div className="grid grid-cols-2 gap-2">
              {attachments.map((attachment) => (
                <div key={attachment.id} className="relative group">
                  <div className="aspect-square rounded-lg overflow-hidden border bg-muted">
                    <img
                      src={attachment.preview}
                      alt={attachment.name}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button
                      size="sm"
                      variant="destructive"
                      className="h-6 w-6 p-0"
                      onClick={() => removeAttachment(attachment.id)}
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                  <div className="mt-1 text-xs text-muted-foreground truncate">
                    {attachment.name}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {formatFileSize(attachment.size)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        <div>
          <label className="text-xs font-medium text-muted-foreground mb-2 block">
            Additional Context (Optional)
          </label>
          <Textarea
            value={contextNotes}
            onChange={(e) => setContextNotes(e.target.value)}
            placeholder="Provide additional context, tone preferences, or specific requirements..."
            className="resize-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
            rows={2}
            disabled={disabled || isLoading}
          />
        </div>
        
        <div className="flex justify-end">
          <Button
            onClick={handleSubmit}
            disabled={(!text.trim() && attachments.length === 0) || isLoading || disabled}
            size="lg"
            className="px-6"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Translating...
              </>
            ) : (
              <>
                <Send className="h-4 w-4 mr-2" />
                {attachments.length > 0 && 'Translate'}
              </>
            )}
          </Button>
        </div>
        
        <div className="text-xs text-muted-foreground">
          <p>💡 <strong>Tips:</strong></p>
          <ul className="mt-1 space-y-1 ml-4">
            <li>• Text input is optional - images alone work fine</li>
            <li>• Paste images in the text box or image upload area (Ctrl+V)</li>
            <li>• Drag and drop images anywhere on this form</li>
            <li>• Be specific about tone (formal, casual, friendly)</li>
            <li>• Mention target audience (gamers, professionals, etc.)</li>
            <li>• Press Enter to translate, Shift+Enter for new line</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  )
}
