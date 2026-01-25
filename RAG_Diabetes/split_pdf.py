import os
from pypdf import PdfReader, PdfWriter

def split_pdf(input_pdf_path, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    reader = PdfReader(input_pdf_path)

    for page_number, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)

        output_path = os.path.join(
            output_folder,
            f"page_{page_number}.pdf"
        )

        with open(output_path, "wb") as output_file:
            writer.write(output_file)

        print(f"Saved: {output_path}")

if __name__ == "__main__":
    input_pdf = "ALGORITHMIC EXPLORERS HANDBOOK.pdf"          # Replace with your PDF name
    output_dir = "split_pages"        # Folder where pages will be saved

    split_pdf(input_pdf, output_dir)
