import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NexusChat Dashboard",
  description:
    "Ask questions about your team's WhatsApp history and get AI-powered answers with verified sources.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-50 text-gray-900 antialiased">{children}</body>
    </html>
  );
}
