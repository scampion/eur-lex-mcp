import json
import os
from logging import DEBUG
import tqdm

from bs4 import BeautifulSoup

from extract_rdf_metadata import parse_rdf_file

HTML_DIR = "LEG_EN_HTML_20251130_01_00"
METADATA_DIR = "LEG_MTD_20251130_01_00"

# for each dir in HTML_DIR and METADATA_DIR, create a corresponding data structure with the txt of the html and the json of the metadata
# example structure:
#
# ├── LEG_EN_HTML_20251130_01_00
# │   ├── 00000dbc-76cd-11ed-9887-01aa75ed71a1
# │   │   └── xhtml
# │   │       └── L_2022316EN.01008601.doc.html
# │   ├── 00012784-19a8-4f6a-8457-fc6291daf9cd
# │   │   ├── html
# │   │   │   └── 31986D0559en.html
# │   │   └── xhtml
# │   │       └── L_1986328EN.01009801.doc.html
# │   ├── 00014bcc-012a-4b01-8fc4-ad58988bceb9
# │   │   └── xhtml
# │   │       └── C_2008203EN.01000401.doc.html
# │   ├── 00064e3d-e914-42e1-a764-73be0b2ea7c5
# │   │   └── xhtml
# │   │       └── L_2010209EN.01000101.doc.html
# │   ├── 0009141e-001e-4c6c-bc03-c60b496a456a
# │   │   └── html
# │   │       └── 32001D0539en.html
# │   ├── 0009a44b-f735-11e9-8c1f-01aa75ed71a1

# For the metadata dir
# LEG_MTD_20251130_01_00
# ├── 00000dbc-76cd-11ed-9887-01aa75ed71a1
# │   └── tree_non_inferred.rdf
# ├── 00012784-19a8-4f6a-8457-fc6291daf9cd
# │   └── tree_non_inferred.rdf
# ├── 0001386d-4004-11ed-92ed-01aa75ed71a1
# │   └── tree_non_inferred.rdf
# ├── 00014bcc-012a-4b01-8fc4-ad58988bceb9
# │   └── tree_non_inferred.rdf
# ├── 00064e3d-e914-42e1-a764-73be0b2ea7c5
# │   └── tree_non_inferred.rdf
# ├── 00086d5c-96dd-43d3-af60-e6dcd8e8a83f
# │   └── tree_non_inferred.rdf
# ├── 0009141e-001e-4c6c-bc03-c60b496a456a
# │   └── tree_non_inferred.rdf
# ├── 0009a44b-f735-11e9-8c1f-01aa75ed71a1
# │   └── tree_non_inferred.rdf
# ├── 000a5af3-6d13-11f0-bf4e-01aa75ed71a1
# │   └── tree_non_inferred.rdf
# ├── 000dfcd5-4ce7-445e-8cda-5d0d95411625

FOUND = 0

with open("eurlex.jsonl", "w", encoding="utf-8") as out_f:
    for dirpath, dirnames, filenames in os.walk(HTML_DIR):
        for filename in filenames:
            if filename.endswith(".html"):
                try:
                    file_id = dirpath.split(os.sep)[-1]
                    subdir = dirpath.split(os.sep)[-2]
                    print(f"Processing {subdir}/{filename}")
                    # read html content
                    filepath = os.path.join(dirpath, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    # convert html to txt with beautifulsoup
                    soup = BeautifulSoup(html_content, "html.parser")
                    text_content = soup.get_text(separator="\n")
                    #remove multiple newlines by replacing them with a single newline
                    text_content = "\n".join([line.strip() for line in text_content.splitlines() if line.strip() != ""])
                    # load corresponding metadata
                    metadata_path = os.path.join(METADATA_DIR, subdir, "tree_non_inferred.rdf")
                    print("Loading metadata from " + metadata_path)
                    # use the package extract_rdf_metadata to extract metadata
                    metadata = parse_rdf_file(metadata_path)

                    # parse item like  metadata.resources.items[21]["owl:sameAs"].local_name and compare against filename.replace("/", ".")
                    local_name = os.path.join(dirpath.split(os.sep)[-1], filename).replace("/", ".")
                    print("Looking for local_name: " + local_name)
                    for item in metadata['resources']['items']:
                        IS_ITEM = False
                        if 'owl:sameAs' in item:
                            if isinstance( item['owl:sameAs'], list):
                                for sameas in item['owl:sameAs']:
                                    if "local_name" in sameas:
                                        if sameas['local_name'].endswith(local_name):
                                            print("✅ Match found between HTML file and RDF metadata for " + filename)
                                            FOUND += 1
                                            IS_ITEM = True
                                            break
                            elif isinstance( item['owl:sameAs'], dict):
                                if "local_name" in item['owl:sameAs']:
                                    if item['owl:sameAs']['local_name'].endswith(local_name):
                                        print("✅ Match found between HTML file and RDF metadata for " + filename)
                                        FOUND += 1
                                        IS_ITEM = True
                                        break
                        if IS_ITEM:
                            uri = item.get('uri', 'N/A')
                            print("  URI: " + uri)
                            record = {
                                "file_id": subdir,
                                "filename": filename,
                                "text": text_content,
                                "uri": uri,
                            }
                            out_f.write(json.dumps(record) + "\n")
                except Exception as e:
                    print(f"❌ Error processing {subdir}/{filename}: {e}")
