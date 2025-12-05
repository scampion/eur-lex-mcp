import requests

resources = [
    "caselaw",
    "decision",
    "directive",
    "intagr",
    "proposal",
    "recommendation",
    "regulation",
]

# create data directory if it doesn't exist
import os
if not os.path.exists('data'):
    os.makedirs('data')
lang = "en"
for resource in resources:
    url = f"https://huggingface.co/datasets/joelito/eurlex_resources/resolve/main/data/{lang}/{resource}.jsonl.xz"
    # if file exists, skip
    if os.path.exists(f"data/{resource}_{lang}.jsonl.xz"):
        print(f"Skipping {resource} in {lang}, file already exists.")
        continue
    response = requests.get(url)
    with open(f"data/{resource}_{lang}.jsonl.xz", "wb") as f:
        f.write(response.content)
    print(f"Downloaded {resource} in {lang}")

# open en caselaw
import lzma
import json
with lzma.open("data/caselaw_en.jsonl.xz", "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        data = json.loads(line)
        print(json.dumps(data, indent=2))

# Example output:
example_output = """
{
  "celex": "62010CN0362",
  "language": "en",
  "date": "2010-07-20",
  "title": "Case C-362/10: Action brought on 20\u00a0July 2010 \u2014 European Commission v Republic of Poland",
  "text": "\n   \n               6.11.2010\u00a0\u00a0\u00a0\n            \n            \n               EN\n            \n            \n               Official Journal of the European Union\n            \n            \n               C 301/4\n            \n         Action brought on 20 July 2010 \u2014 European Commission v Republic of Poland\n   (Case C-362/10)\n   ()\n   2010/C 301/06\n   Language of the case: Polish\n   \n      Parties\n   \n   \n      Applicant: European Commission (represented by: S. La Pergola and K. Herrmann, acting as Agents)\n   \n      Defendant: Republic of Poland\n   \n      Form of order sought\n   \n   \n               \u2014\n            \n            \n               declare that, by failing to adopt all the laws and regulations necessary for the proper transposition of Articles 2, 3, 4, 6, 7, 8, 10 and 11 of Directive 2003/98/EC of the European Parliament and of the Council of 17 November 2003 on the re-use of public sector information,\u00a0(1) the Republic of Poland has failed to fulfil its obligations under those provisions of the directive;\n            \n         \n               \u2014\n            \n            \n               order the Republic of Poland to pay the costs.\n            \n         \n      Pleas in law and main arguments\n   \n   In the applicant\u2019s submission, the Republic of Poland has hitherto not adopted national measures correctly transposing Directive 2003/98 into national law. The Ustawa z 6 wrze\u015bnia 2001 r. o dost\u0119pie do informacji publicznej (Law of 6 September 2001 on access to public information), which was notified to the Commission, does not relate to the re-use of public sector information, because it does not even contain a definition of \u2018re-use\u2019. For that reason alone, the rights and obligations resulting from that Law cannot constitute a correct transposition of Directive 2003/98.\n   \n      (1)\u00a0\u00a0OJ 2003 L 345, p. 90.\n   "
}
{
  "celex": "62015TA0385",
  "language": "en",
  "date": "2016-06-14",
  "title": "Case T-385/15: Judgment of the General Court of 14 June 2016 \u2014 Loops v EUIPO (Shape of a toothbrush) (EU trade mark \u2014 International registration designating the European Union \u2014 Three-dimensional mark \u2014 Shape of a toothbrush \u2014 Absolute ground for refusal \u2014 Lack of distinctive character \u2014 Article 7(1)(b) of Regulation (EC) No 207/2009)",
  "text": "\n   \n               25.7.2016\u00a0\u00a0\u00a0\n            \n            \n               EN\n            \n            \n               Official Journal of the European Union\n            \n            \n               C 270/41\n            \n         Judgment of the General Court of 14\u00a0June 2016\u00a0\u2014 Loops v EUIPO (Shape of a toothbrush)\n   (Case T-385/15)\u00a0(1)\n   \n   ((EU trade mark - International registration designating the European Union - Three-dimensional mark - Shape of a toothbrush - Absolute ground for refusal - Lack of distinctive character - Article\u00a07(1)(b) of Regulation (EC) No\u00a0207/2009))\n   (2016/C 270/46)\n   Language of the case: English\n   \n      Parties\n   \n   \n      Applicant: Loops, LLC (Dover, Delaware, United States) (represented by: T.\u00a0Schmidpeter, lawyer)\n   \n      Defendant: European Union Intellectual Property Office (represented by: W.\u00a0Schramek and A.\u00a0Schifko, acting as Agents)\n   \n      Re:\n   \n   Action brought against the decision of the Second Board of Appeal of EUIPO of 30\u00a0April 2015 (Case\u00a0R 1917/2014-2), concerning an application for registration of a three-dimensional sign consisting of the shape of a toothbrush as an EU trade mark.\n   \n      Operative part of the judgment\n   \n   The Court:\n   \n               1.\n            \n            \n               Dismisses the action;\n            \n         \n               2.\n            \n            \n               Orders Loops, LLC, to pay the costs.\n            \n         \n      (1)\u00a0\u00a0OJ C\u00a0302, 14.9.2015.\n   "
}
"""
