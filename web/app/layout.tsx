import type { Metadata } from 'next';
import './globals.css';
// Only a deployment-owned configuration may set the origin; never use request host headers.
const siteOrigin = new URL(process.env.SITE_URL || 'http://localhost:3000');
export const metadata: Metadata = {
  metadataBase: siteOrigin,
  title: '主动伴护 · 视觉与环境控制台',
  description:
    '认知症主动伴护演示原型：本机 YOLO 姿态识别、连续动作候选与模拟环境事件处理。',
  openGraph: {
    title: '主动伴护 · 视觉与环境控制台',
    description: '观察有依据，干预有边界。工业设计演示原型。',
    images: [
      {
        url: new URL('/og.png', siteOrigin).href,
        width: 1536,
        height: 1024,
        alt: '主动伴护：观察有依据，干预有边界',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: '主动伴护 · 视觉与环境控制台',
    description: '本机 YOLO 姿态识别、连续动作候选与模拟环境事件处理。',
    images: [new URL('/og.png', siteOrigin).href],
  },
};
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
