"""Granite-Docling writes DocTags, a markup with page positions; this turns it into Markdown."""

import cv2 as cv


def doctags_to_markdown(doctags, image):
    """Markdown of one page. Needs docling-core; without it the DocTags come back unchanged."""
    try:
        from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
        from PIL import Image
    except ImportError:
        return doctags
    page = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))   # used to crop pictures
    pairs = DocTagsDocument.from_doctags_and_image_pairs([doctags], [page])
    return DoclingDocument.load_from_doctags(pairs, document_name="page").export_to_markdown()
