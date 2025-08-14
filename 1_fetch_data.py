import arxiv
import os

# Create a directory to store PDFs if it doesn't exist
if not os.path.exists('papers'):
    os.makedirs('papers')

# Search for papers in the Quantitative Biology category
search = arxiv.Search(
  query = "cat:q-bio.GN", # GN = Genomics
  max_results = 10, # Let's start with 10 papers for our MVP
  sort_by = arxiv.SortCriterion.SubmittedDate
)

# Download the PDF for each result
for result in search.results():
    try:
        # The filename will be the paper's entry_id
        filename = f"{result.entry_id.split('/')[-1]}.pdf"
        filepath = os.path.join('papers', filename)
        result.download_pdf(dirpath="./papers", filename=filename)
        print(f"Downloaded: {result.title} to {filepath}")
    except Exception as e:
        print(f"Failed to download {result.title}. Reason: {e}")