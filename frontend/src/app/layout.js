import './globals.css'

export const metadata = {
  title: 'Ollama Chat',
  description: 'A beautiful chatbot interface powered by Ollama',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
