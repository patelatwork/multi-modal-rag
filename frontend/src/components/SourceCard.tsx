import { useState } from "react";
import type { Source } from "../types";
import { ImageLightbox } from "./ImageLightbox";
import { Markdown } from "./Markdown";

const KIND_ICON: Record<Source["kind"], string> = {
  figure: "▨",
  table: "▦",
  text: "¶",
};

/**
 * One retrieved source.
 *
 * For a figure or table this shows the actual crop the model was given, which
 * is the point of the whole system: the user can check the answer against the
 * chart rather than trusting a paraphrase of it.
 */
export function SourceCard({ source }: { source: Source }) {
  const [open, setOpen] = useState(false);
  const [zoomed, setZoomed] = useState(false);

  const page = source.page_number ? `page ${source.page_number}` : null;
  const score = source.score !== null ? source.score.toFixed(2) : null;
  const altText = source.caption ?? `${source.kind} from ${page ?? "document"}`;

  return (
    <div className={`source-card source-${source.kind}`}>
      <button className="source-head" onClick={() => setOpen((value) => !value)}>
        <span className="source-icon" aria-hidden="true">
          {KIND_ICON[source.kind]}
        </span>
        <span className="source-label">{source.label}</span>
        <span className="source-kind">{source.kind}</span>
        {page && <span className="source-page">{page}</span>}
        {score && <span className="source-score">{score}</span>}
        <span className="source-toggle">{open ? "−" : "+"}</span>
      </button>

      {source.caption && <p className="source-caption">{source.caption}</p>}

      {source.image_url && (
        <>
          <img
            className="source-image"
            src={source.image_url}
            alt={altText}
            loading="lazy"
            onClick={() => setZoomed(true)}
          />
          {zoomed && (
            <ImageLightbox
              src={source.image_url}
              alt={altText}
              onClose={() => setZoomed(false)}
            />
          )}
        </>
      )}

      {open && (
        <div className="source-body">
          {source.table_markdown ? (
            <Markdown>{source.table_markdown}</Markdown>
          ) : (
            <p className="source-text">{source.text}</p>
          )}
        </div>
      )}
    </div>
  );
}
