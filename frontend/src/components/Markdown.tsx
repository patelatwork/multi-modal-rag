import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";

/**
 * Answers routinely contain Markdown tables (the system prompt asks for them
 * when comparing values), so GFM support is required rather than cosmetic.
 * Math is rendered too, because scientific PDFs carry units and symbols like
 * $\mu g/m^3$ straight through into the answer.
 *
 * Tables are wrapped so a wide one scrolls inside the bubble instead of
 * stretching the whole page.
 */
export function Markdown({ children }: { children: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }]]}
        components={{
          table: ({ node: _node, ...props }) => (
            <div className="table-scroll">
              <table {...props} />
            </div>
          ),
          a: ({ node: _node, ...props }) => (
            <a {...props} target="_blank" rel="noreferrer noopener" />
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
