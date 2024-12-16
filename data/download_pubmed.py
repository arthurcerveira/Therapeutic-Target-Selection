from datasets import load_dataset
import os


pubmed = load_dataset('ncbi/pubmed', streaming=True)
dir_idx = 0
count = 0

for entry in pubmed['train']:
    pmid = entry['MedlineCitation']['PMID']

    if os.path.exists(f'./pubmed/{dir_idx}/{pmid}.txt'):
        continue

    abstract_text = entry['MedlineCitation']['Article']['Abstract']['AbstractText']
    abstract_title = entry['MedlineCitation']['Article']['ArticleTitle']

    if not abstract_text or not abstract_title:
        continue

    text = abstract_title + '\n' + abstract_text
    with open(f'./pubmed/{dir_idx}/{pmid}.txt', 'w') as f:
        f.write(text)

    count += 1

    if count % 100_000 == 0 and count != 0:
        print(f'{count} entries processed')
        dir_idx += 1
        print(f'Creating new directory: {dir_idx}')
        # Create a new directory for the next batch of files
        os.makedirs(f'./pubmed/{dir_idx}', exist_ok=True)

    if count == 30_000_000:
        break
