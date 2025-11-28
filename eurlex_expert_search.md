# Expert Search on EUR-Lex

## Choose the Experimental Features You Want to Try

This document is an excerpt from the EUR-Lex website.

Use quotation marks to search for an "exact phrase". Append an asterisk (*) to a search term to find variations of it (transp*, 32019R*). Use a question mark (?) instead of a single character in your search term to find variations of it (ca?e finds case, cane, care).

Need more search options? Use the [Advanced search](https://eur-lex.europa.eu/content/help/search/advanced-search-form.html).

## How to Perform an Expert Search

Make complex searches by:

1. Signing in with your My EUR-Lex account.
2. Clicking Expert search in the top right corner of your screen.

### Steps

1. Select your Search language.
2. Enter your search parameters in the Expert query box, using the appropriate syntax, in one of 2 ways:
   1. Type your query manually in the box.
   2. Use the system prompts.
   3. Click an item in Search fields and the system will add the relevant field to the query box, as well as displaying (on the right) information and the appropriate help.

   **Codes or Full Names**  
   By default search fields are entered in the query box as a code (e.g. TI). To display search fields with their full name (e.g. Title), select the radio button Text, below the box.  
   Your query can contain a mix of codes for some fields and full names for others.

   The search fields (and their explanation) are only visible if JavaScript is enabled in your browser.

   You can also use the blue buttons in the toolbox to the right of the query box to add search operators, wildcards or field specifiers.

   See more in the next chapter: Syntax & search operators.

3. Click Search.

If your syntax is invalid, the system will display an error message. You can also manually check if your syntax is valid by pressing the Check syntax button.

To empty all fields, click Clear.

## Syntax & Search Operators

Your expert search query must contain at least the following 3 elements (in this order):

| Search Field (Metadata) | Field Specifier | Search Term |
|-------------------------|-----------------|--------------|
| Example: Title          | ~               | Europa       |

You can create more complex queries by using search operators, like AND.

**Example**

| Title  | ~     | Europa | AND | Author | =   | COM  |
|--------|-------|--------|-----|--------|-----|------|
| Search Field | Field Specifier | Search Term | Search Operator | Search Field | Field Specifier | Search Term |

### Operators

For more complex queries. Type them in manually or use the blue buttons on the right side of your screen.

| Operator | Function |
|----------|----------|
| AND      | Finds results that contain all your search terms |
| OR       | Finds results that contain at least one of your search terms |
| NOT      | Excludes search terms from your results |
| WHEN     | Searches for a subset of a larger parent group |
| ()       | Groups search terms and specifies the order in which they are interpreted |
| NEAR10   | Finds search terms within 10 words of each other |
| NEAR40   | Finds search terms within 40 words of each other |
| *        | Substitutes for 0 to n characters |
| ?        | Substitutes for a single character |
| <        | Less than |
| <=       | Less than or equal to |
| >        | Greater than |
| >=       | Greater than or equal to |
| <>       | Not equal to |
| =        | Equal to; to be used only for Coded data when searching for an exact match of the search term and result |
| ~        | Contains; to be used only for Text data when searching for documents containing your search term |

### Searching for Specific Terms

**Query** | **Finds**
---|---
By default, EUR-Lex will search for all words derived from the basic root word you enter. Text ~ equal | Documents containing the term 'equal', but also e.g. 'equality' or 'equalization' in the text.
To search only for the precise term, use quotations marks. Text ~ "equal" | Documents containing only the term 'equal' in the text.
AND | Finds results that contain all your search terms. Text ~ personal AND data | Documents containing the words 'personal' and 'data' in the text.
Title ~ "equal treatment" AND Text ~ fundamental | Documents with the words "equal treatment" in the title and the words "fundamental" in the text.
Title ~ "personal data" AND Collection = LEGISLATION | Legislation containing the phrase 'personal data' in the title.

**OR**  
Finds results that contain at least 1 of your search terms:  
Title ~ "equal treatment" OR Text ~ fundamental | Documents that have either the words "equal treatment" in the title or the word "fundamental" in the text.

**NOT**  
Excludes search terms from your results:  
Title ~ "equal treatment" NOT Text ~ fundamental | Documents that have the words "equal treatment" in the title, but which do not have the word "fundamental" in the text.

- You cannot combine a full text search with a search on other Search fields (metadata) by using the OR operator.  
  - Title ~ "personal data" OR Collection = LEGISLATION is not a valid query.  
  - To combine these 2 elements of your search, you would instead have to use either the AND or NOT operators.  
  - Title ~ "personal data" AND Collection = LEGISLATION is a valid query.  
  - Title ~ "personal data" NOT Collection = LEGISLATION is a valid query.

### Order of Precedence of AND, OR and NOT Operators

There is a fixed order of precedence between these operators. 'NOT' has precedence over 'AND', and 'AND' has precedence over 'OR'.

**Example**

| Query | Finds |
|-------|-------|
| Title ~ equal AND treatment NOT rights OR Text ~ fundamental | Documents whose title contains: <br> - both 'equal' AND 'treatment', but NOT 'rights' <br> - OR <br> - 'fundamental' in the text <br> To specify a different priority, use brackets. |

### BRACKETS ( )

You can use brackets to group elements in your query and specify the order in which they will be interpreted – even to change the default priority order between AND, NOT and OR.

| Query | Finds |
|-------|-------|
| NO BRACKETS: Title ~ equal AND treatment OR rights | Documents whose title contains: <br> - both 'equal' AND 'treatment' <br> OR <br> - only the word 'rights' |
| WITH BRACKETS ( ): Title ~ equal AND (treatment OR rights) | Documents whose title contains either: <br> - 'equal' and 'treatment' <br> OR <br> - 'equal' and 'rights' |

### WHEN

Use this to find a subset of a larger parent group. WHEN functions similarly to AND, but is more restrictive.

**Example**

| Query | Finds |
|-------|-------|
| Legal_basis = 32012R1151 WHEN Legal_basis_paragraph = 3 | Documents: <br> - whose legal basis is the document with CELEX number 32012R1151, and <br> - which are based on paragraph 3 of that document. |

### NEAR10

Finds terms within 10 words of a specified other term.

**Example**

| Query | Finds |
|-------|-------|
| Text ~ "transport" NEAR10 "perishable" | Documents containing the words 'transport' and 'perishable' within 10 words of each other (regardless of sentences and paragraphs). |

### NEAR40

Functions as NEAR10 but within 40 words of a specified term.

You can only use NEAR10 and NEAR40 for text searches.

### WILDCARDS (* and ?)

| Query | Finds |
|-------|-------|
| * | Substitutes for 0 to n characters. expe* will find expert, expenditure, expectation, etc. |
| ? | Substitutes for a single character. ca?e will find case, care, cane, etc. |

Wildcards cannot be used to begin a search term. Text ~ *tation is not a valid query.

### RANGES

To enter a range of values for dates or numbers, use the following operators:

| Query | Finds |
|-------|-------|
| < (less than) | For dates, this means "before" e.g. Date_of_publication < 01/01/2015 |
| <= (less than or equal to) | |
| > (greater than) | |
| >= (greater than or equal to) | |
| <> (not equal to) | |

### EQUALS (=)

Finds results that match your search term exactly. The operator is automatically added to the appropriate fields when chosen from Search fields.

Note: the equals operator cannot be used with text data. E.g. fields for dates or numbers will return results for that exact date or number.

**Query**

| Query | Finds |
|-------|-------|
| Date_of_publication = 16/11/2010 | Documents published on 16/11/2010 only. |

### TILDE (~)

Finds all texts containing the search term you enter.

**Query**

| Query | Finds |
|-------|-------|
| Title ~ Commission | Documents whose title contains the word 'Commission'. |

## Search Fields

The search fields are displayed below the expert search box, on the left side. These are all the fields you can search on, grouped into categories. By default only the top levels are shown. Click on them to display or hide the sub-levels.

## Help for the Search Fields

When you click an item in the list of search fields, the system displays (to the right of the list) explanations, examples and tips on how to use that field:

### Values (List Form)

For some types of fields (e.g. Documents –> Coded data –> Classifications –> Subject_matter (CT)), the system displays a list of possible values that you can select for your search (check the box in front of your desired value(s) and click the Add to query button). To find a specific value – type it in the Filter box at the top of the list and click the Filter button.

### Values (Tree Form)

Some values are displayed in tree form (e.g. Documents –> Coded data –> Classifications –> EUROVOC_descriptor (DC)). Click to expand an item and check the box in front of your desired value(s) to select it (you can select multiple values). To find a specific value – type it in the Filterbox at the top of the list and click the Filter button. Click the Add to query button (multiple values will be separated by 'OR').

### Search by Date

Search by an exact date or a range of dates:  
1. Select a date-related field (e.g. Documents –> Coded data –> Date –> Date_of_publication) and select from the dropdown on the right either:  
   a. Exact date  
   b. Range of date  
2. Type the dates manually (DD/MM/YYYY) or use the calendar button.  
3. Click the Add to query button.

### Relationships Between Documents

To search for documents by relationship (e.g. find a document amended by another, or a document affected by a certain case):  
1. Click Documents –> Coded data –> Relationship between documents.  
2. Click a type of relationship, e.g. Amended_by (MD).  
3. On the right you will see:  
   a. A field for the CELEX number. If the CELEX number contains brackets, please put it in quotation marks (e.g. "42000A0922(02)".  
   b. In some cases also a dropdown for the Primary role, where you can be much more specific about the relationship.  
4. Click one of the values in the results list.  
5. Click the Add to query button.  

The fields for Legal Basis allow you to specify your search on a specific article, paragraph or even subparagraph.

### Comments on the Relationships Between Documents

You can also search on the comments made to relationships between documents. Examples of fields with comments on relationships between documents include: Documents –> Coded data –> Relationship between documents –> Case_affecting (AJ); Instruments_cited (CI); or Legal_basis (LB). To search on comments made on relationships between documents see procedure described above in "Relationships between documents".

## Personalise Your Search Results

Below the Expert query box, you can use the Customise shown information button to change how your results are displayed.

### Highlighting Words in Your Results – Text ZOOM

To highlight – in your results list – words from any text fields you are searching with:  
1. Click Text ZOOM (under Metadata to display).  
2. Select the number of words to be displayed before/after your search terms.  
To apply all these changes to your search results, click Apply.