# open the pdf file in ./pdfs and parse the text
# save the text in ./txts
import pdftotext

# Load your PDF
with open("pdf/scania-annual-and-sustainability-report-2022.pdf", "rb") as f:
    pdf = pdftotext.PDF(f)
    # save text in pdf to file in ./txt
    with open("txt/scania-annual-and-sustainability-report-2022.txt", "w") as f:
        f.write("\n\n".join(pdf))
