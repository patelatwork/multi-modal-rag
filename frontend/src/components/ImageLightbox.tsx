import { useEffect } from "react";
import { createPortal } from "react-dom";

interface Props {
  src: string;
  alt: string;
  onClose: () => void;
}

/**
 * Full-screen preview for a source crop.
 *
 * Rendered in a portal with its own backdrop so it cannot be trapped inside a
 * scrolling message bubble, and closes on Escape as well as click — an overlay
 * with no keyboard escape hatch is a trap for keyboard and screen-reader users.
 */
export function ImageLightbox({ src, alt, onClose }: Props) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previousOverflow;
    };
  }, [onClose]);

  return createPortal(
    <div className="lightbox" role="dialog" aria-modal="true" aria-label={alt} onClick={onClose}>
      <img src={src} alt={alt} onClick={(event) => event.stopPropagation()} />
      <button className="lightbox-close" onClick={onClose} aria-label="Close preview">
        ×
      </button>
    </div>,
    document.body,
  );
}
