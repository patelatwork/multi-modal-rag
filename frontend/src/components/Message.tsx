import type { ChatMessage } from "../types";
import { Markdown } from "./Markdown";
import { SourceCard } from "./SourceCard";

export function Message({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className="message user">
        <div className="bubble">{message.content}</div>
      </div>
    );
  }

  const visuals = message.sources?.filter((source) => source.image_url) ?? [];
  const textual = message.sources?.filter((source) => !source.image_url) ?? [];

  return (
    <div className="message assistant">
      <div className={`bubble${message.error ? " error" : ""}`}>
        <Markdown>{message.content}</Markdown>

        {visuals.length > 0 && (
          <section className="sources">
            <h4>
              Figures &amp; tables
              {message.meta && message.meta.images > 0 && (
                <span className="sources-note">
                  {message.meta.images} sent to the model
                </span>
              )}
            </h4>
            <div className="source-grid">
              {visuals.map((source) => (
                <SourceCard key={source.id} source={source} />
              ))}
            </div>
          </section>
        )}

        {textual.length > 0 && (
          <section className="sources">
            <h4>Passages</h4>
            <div className="source-grid">
              {textual.map((source) => (
                <SourceCard key={source.id} source={source} />
              ))}
            </div>
          </section>
        )}

        {message.meta && (
          <div className="meta">answered in {message.meta.seconds.toFixed(1)}s</div>
        )}
      </div>
    </div>
  );
}
