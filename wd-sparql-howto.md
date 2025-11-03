# Wikidata SPARQL Composition: A Part-to-Whole Strategy Guide

## Overview

This guide describes a proven strategy for composing SPARQL queries against Wikidata, a large-scale knowledge graph containing millions of entities and relationships. Rather than attempting to write complete queries from scratch, this approach builds queries incrementally through exploration and validation.

The core insight: **Start simple, validate often, and grow gradually.**

## The Part-to-Whole Philosophy

Traditional query composition often fails when dealing with unfamiliar knowledge bases because assumptions about structure and naming conventions are frequently incorrect. The part-to-whole strategy addresses this by:

1. **Breaking down** the query construction into small, testable fragments
2. **Validating** each fragment against the actual knowledge base
3. **Learning** from the knowledge base structure itself
4. **Building up** to the complete query through confirmed steps

Think of it like exploring a new city: you don't assume you know all the street names and routes. Instead, you consult maps, ask locals, and verify each turn before proceeding.

## Core Principles

### 1. Start with Simple Fragments

Begin with the most basic query possible. Don't try to express the full question in your first attempt.

**Example scenario:** "Which artists were born on the same date as Rachel Stevens?"

**Bad approach:** Try to write the complete query immediately
```
SELECT ?artist ?birthDate WHERE {
  ?rachel <hasName> "Rachel Stevens".
  ?rachel <bornOn> ?date.
  ?artist <bornOn> ?date.
  ?artist <hasOccupation> <Artist>.
}
```
This assumes property names, structure, and identifiers that may not exist.

**Good approach:** Start by finding Rachel Stevens
```
Find entity: "Rachel Stevens"
Result: Entity Q6108160
```

### 2. Validate Assumptions Before Proceeding

Never assume you know how the knowledge base is structured. Always verify.

**After finding Rachel Stevens (Q6108160):**
- Don't assume the birth date property is called "bornOn"
- Instead, explore what properties this entity actually has
- Discover that birth date is stored under property P569
- Confirm the exact date value: 1978-04-09

### 3. Explore the Knowledge Base Structure

Use exploration tools to understand how information is organized:

**Entity Exploration:** "What properties does Rachel Stevens have?"
- Returns: birth date (P569), occupation (P106), place of birth (P19), etc.

**Property Exploration:** "How is occupation typically used?"
- Shows examples of how P106 connects people to their professions
- Reveals that "artist" might be represented as entity Q483501

**Search Capabilities:** "What's the identifier for 'artist'?"
- Search for concept: "artist"
- Find: Q483501 represents "artist" as an occupation

### 4. Build Incrementally

Add one constraint at a time, testing after each addition:

**Step 1:** Find people born on the same date
```
Query: All entities with birth date 1978-04-09
Test: Returns results
```

**Step 2:** Add occupation filter
```
Query: Entities born 1978-04-09 AND have occupation "artist"
Test: Returns smaller, filtered set
```

### 5. Avoid Repetition

If you've already tried an exploration step or query and received results, don't repeat it. Use what you've learned to move forward.

**Wasteful:**
- Search for "Rachel Stevens" → Get Q6108160
- Search for "Rachel Stevens" again → Get Q6108160 (no new information)

**Efficient:**
- Search for "Rachel Stevens" → Get Q6108160
- Explore Q6108160 properties → Learn structure
- Build query using discovered structure

### 6. Request Entities, Not Just Labels

When retrieving results, request the actual entity identifiers, not just human-readable labels. This allows for further exploration and verification.

**Less useful:** Return only "Mountain Time Zone" (just text)
**More useful:** Return "Mountain Time Zone (Q2134)" (entity with identifier)

The identifier enables follow-up queries like "What other cities use this time zone?"

## The Iterative Workflow

A typical query composition follows this pattern:

### Phase 1: Entity Discovery
1. Identify key entities mentioned in the question
2. Search for their identifiers in the knowledge base
3. Confirm you have the correct entities (disambiguation)

**Example:** "What is the time zone of Salt Lake City?"
- Search: "Salt Lake City" → Multiple results
- Disambiguate: City in Utah, USA → Q23337
- Verify: Check that Q23337 is indeed the Utah city

### Phase 2: Structure Exploration
1. Explore the discovered entities
2. Identify relevant properties
3. Understand how values are represented

**Continuing example:**
- Explore Q23337 (Salt Lake City)
- Find property P421 (time zone)
- Note that time zones are represented as entities themselves

### Phase 3: Fragment Testing
1. Write a simple query fragment
2. Execute to verify it works
3. Examine results to confirm correctness

**Test fragment:**
```
Get all properties of Salt Lake City
Verify: Returns many properties including P421
```

### Phase 4: Incremental Building
1. Add constraints one at a time
2. Test after each addition
3. Adjust based on results

**Build up:**
```
Version 1: Get time zone property value
Version 2: Include human-readable labels
Version 3: Filter to distinct results only
```

### Phase 5: Validation and Refinement
1. Execute the complete query
2. Verify results answer the original question
3. Optimize if needed (remove redundancy, improve efficiency)

## Common Patterns

### Pattern 1: Simple Property Lookup

**Question type:** "What is X's Y?"

**Strategy:**
1. Find entity X
2. Explore X's properties
3. Identify property for Y
4. Query X's value for that property

**Example:** "What is Frank Herbert's profession?"
- Find: Frank Herbert → Q7934
- Explore: Q7934 properties
- Find: P106 (occupation)
- Query: Q7934's P106 values

### Pattern 2: Shared Property Values

**Question type:** "What shares property Y with X?"

**Strategy:**
1. Find entity X
2. Get X's value for property Y
3. Query all entities with same Y value
4. Filter to relevant types if needed

**Example:** "Which artists were born on the same date as Rachel Stevens?"
- Find: Rachel Stevens → Q6108160
- Get: Q6108160's birth date → 1978-04-09
- Query: All entities with birth date 1978-04-09
- Filter: Only those with occupation = artist

### Pattern 3: Property Chain Traversal

**Question type:** "What is Y of X's Z?"

**Strategy:**
1. Find entity X
2. Get X's Z values
3. For each Z, get its Y values
4. Aggregate results

**Example:** "What languages are spoken in countries bordering France?"
- Find: France → Q142
- Get: Countries bordering Q142 (via P47)
- For each bordering country, get official languages (via P37)
- Collect all distinct languages

### Pattern 4: Filtering by Multiple Constraints

**Question type:** "What X has property Y and property Z?"

**Strategy:**
1. Identify the entity type X
2. Add first constraint Y
3. Test and verify results
4. Add second constraint Z
5. Continue adding constraints incrementally

**Example:** "Which American presidents were in office during the Vietnam War?"
- Constraint 1: Position held = President of the United States
- Test: Returns all US presidents
- Constraint 2: Time period overlaps 1955-1975
- Test: Returns presidents during that era

## Error Recovery Strategies

### Syntax Errors
**Signal:** Query won't execute, syntax error reported

**Response:**
- Review the query structure
- Check property naming (P### format)
- Verify entity references (Q### or wd:Q### format)
- Ensure proper punctuation and brackets
- Simplify to a working fragment and rebuild

### No Results
**Signal:** Query executes but returns empty set

**Possible causes:**
1. Wrong property identifier
2. Wrong entity identifier
3. Value format mismatch (e.g., date format)
4. Too restrictive constraints

**Response:**
- Verify each component separately
- Test constraints individually
- Explore entities to confirm property usage
- Check example usage of properties

### Query Timeout
**Signal:** Query takes too long to execute

**Response:**
- Add LIMIT clause to reduce result set
- Make query more specific (add constraints)
- Avoid expensive operations like unbounded property paths
- Consider inverse property formulations

### Too Many Results
**Signal:** Query returns thousands of results when expecting dozens

**Response:**
- Add missing constraints
- Use DISTINCT to remove duplicates
- Verify you're querying the right property
- Consider if the question needs more specificity

## Best Practices

### Do:
- ✓ Explore before querying
- ✓ Test each component separately
- ✓ Build incrementally
- ✓ Request entity identifiers for verification
- ✓ Use DISTINCT when appropriate
- ✓ Include label services for human-readable output
- ✓ Learn from property examples

### Don't:
- ✗ Assume property names or structures
- ✗ Write complete queries without validation
- ✗ Repeat the same exploration steps
- ✗ Skip verification of intermediate results
- ✗ Request only labels (lose verification capability)
- ✗ Add all constraints simultaneously
- ✗ Ignore timeout or error signals

## Query Optimization Tips

Once you have a working query, consider these optimizations:

### 1. Use Property Paths Efficiently
Prefer direct properties over multi-hop paths when possible. Property paths (like `wdt:P31/wdt:P279*`) are powerful but can be slow.

### 2. Put Selective Constraints First
Order your query constraints from most to least selective. This helps query optimizers narrow the search space early.

### 3. Use DISTINCT Appropriately
Only use DISTINCT when you expect and want to remove duplicates. It adds processing overhead.

### 4. Limit Results When Exploring
When testing fragments, use LIMIT clauses to avoid fetching thousands of results unnecessarily.

### 5. Leverage Label Service
Include the Wikidata label service to get human-readable names without separate lookups:
```
SERVICE wikibase:label {
  bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
}
```

## Real-World Application

Consider a complex question: "Which paintings by Leonardo da Vinci are displayed in museums in France?"

**Part-to-Whole Breakdown:**

**Part 1: Find Leonardo da Vinci**
- Search: "Leonardo da Vinci"
- Result: Q762

**Part 2: Explore his works**
- Get Q762's properties
- Find: P800 (notable works)
- Alternative: Find works that have P170 (creator) = Q762

**Part 3: Filter to paintings**
- Get entity type for "painting"
- Search: "painting" → Q3305213
- Add constraint: works must have P31 (instance of) = Q3305213

**Part 4: Find locations**
- Test: Get location (P276) of these paintings
- Verify: Some have locations, others don't

**Part 5: Filter to France museums**
- Find: France → Q142
- Find: museum type → Q33506
- Add constraint: location must be in France AND be a museum

**Result:** Incrementally built query with verified structure at each step.

## Conclusion

The part-to-whole SPARQL composition strategy succeeds by embracing exploration over assumption. By starting simple, validating frequently, and building incrementally, you can construct accurate queries even for unfamiliar knowledge bases.

Key takeaways:
1. **Explore first** - Understand the knowledge base structure before querying
2. **Validate often** - Test each component as you build
3. **Build incrementally** - Add one constraint at a time
4. **Learn from errors** - Use failures to refine your understanding
5. **Avoid repetition** - Each step should add new information

This approach may seem slower than writing queries directly, but it dramatically increases success rates and reduces debugging time, especially when working with large, complex knowledge bases like Wikidata.

---

## Appendix A: Technical Implementation Details

### System Architecture

The SPARQL composition system is implemented as a state machine with the following components:

**Core Components:**
- **State Manager** (`parser_state.py`): Tracks conversation history, generated queries, and execution results
- **Parser Engine** (`part_to_whole_parser.py`): Implements the state machine with 7 nodes
- **Utility Functions** (`wikidata_utils.py`): Handles SPARQL execution, optimization, and Wikidata API calls
- **Helper Chains** (`parser_utils.py`): Provides LangChain integration for LLM interactions

### State Machine Graph

```
start → controller → [execute_sparql | get_wikidata_entry |
                      search_wikidata | get_property_examples]
                      ↓
                   controller → ... → reporter → END
                      ↓
                    stop
```

**Nodes:**
1. `start`: Initialize state with question
2. `controller`: LLM decides next action based on history
3. `execute_sparql`: Execute SPARQL query against Wikidata endpoint
4. `get_wikidata_entry`: Retrieve entity properties and qualifiers
5. `search_wikidata`: Search for entities/properties by name
6. `get_property_examples`: Get example usage of a property
7. `stop`: Stop exploration and extract final answer
8. `reporter`: Format and return final results

### Key Classes

**SparqlQuery Class** (`parser_state.py`)
```python
class SparqlQuery:
    def __init__(self, sparql: Optional[str] = None):
        self.sparql = SparqlQuery.clean_sparql(sparql)
        self.preverbalized, self.label_to_id_map = SparqlQuery.preverbalize(self.sparql)
        self.execution_result = None
```

Attributes:
- `sparql`: Cleaned SPARQL query string
- `preverbalized`: Human-readable version with labels instead of IDs
- `label_to_id_map`: Mapping of labels to QIDs/PIDs
- `execution_result`: Query execution results and status

**State Class** (`parser_state.py`)
```python
class State:
    question: str
    messages: list[BaseMessage]
    actions: list[str]
    generated_sparqls: list[SparqlQuery]
    num_steps: int
```

Tracks:
- Original question
- Conversation history
- Actions taken (for deduplication)
- All generated SPARQL queries
- Iteration count (max 15)

### Action Types

The controller can choose from 5 actions at each step:

1. **execute_sparql(SPARQL)**
   - Executes query against Wikidata SPARQL endpoint
   - Returns: Results table or error status
   - Timeout: 30 seconds
   - Max result size: 10MB

2. **get_wikidata_entry(QID)**
   - Retrieves all outgoing properties from an entity
   - Includes: Direct properties and qualifiers
   - Filters: Only properties relevant to current question
   - Returns: Property-value pairs with labels

3. **search_wikidata(search_string)**
   - Searches Wikidata for entities/properties matching string
   - Returns: Top 10 matches with QID/PID and descriptions
   - Types: Can search both entities (Q) and properties (P)

4. **get_property_examples(PID)**
   - Retrieves example usage of a property
   - Returns: Sample subject-object pairs using this property
   - Limit: Up to 10 examples
   - Helps understand property semantics

5. **stop()**
   - Signals completion of query construction
   - Triggers final answer extraction
   - Returns control to reporter node

### SPARQL Query Processing Pipeline

**Step 1: Cleaning**
```python
def clean_sparql(sparql: str):
    # Remove comments
    cleaned = re.sub(r"#.*", "", sparql)
    # Apply optimizations
    cleaned = try_to_optimize_query(cleaned)
    # Prettify formatting
    cleaned = prettify_sparql(cleaned)
    return cleaned
```

**Step 2: Preverbalization**
```python
def preverbalize(sparql: str):
    # Replace QIDs with labels: wd:Q762 → Leonardo_da_Vinci_(Q762)
    # Replace PIDs with labels: wdt:P800 → notable_works_(P800)
    # Returns: (preverbalized_sparql, label_to_id_map)
```

**Step 3: Optimization**
```python
def try_to_optimize_query(sparql: str):
    # Inverse property optimization
    # BEFORE: ?x wdt:P31/wdt:P279* ?class
    # AFTER:  ?class ^wdt:P279*/^wdt:P31 ?x

    # COUNT optimization
    # BEFORE: COUNT(?item)
    # AFTER:  COUNT(*)
```

**Step 4: Execution**
```python
def execute_sparql(sparql: str, timeout: int = 30):
    # Send to Wikidata endpoint: https://query.wikidata.org/sparql
    # Parse JSON results
    # Handle errors: syntax, timeout, too large
    # Return: (results_table, execution_status)
```

### Error Handling

**Exception Types:**
```python
class SPARQLSyntaxError(Exception)          # Invalid syntax
class SPARQLTimeoutError(Exception)         # Query timeout
class SPARQLResultsTooLargeError(Exception) # Result > 10MB

class SparqlExecutionStatus(Enum):
    OK = 0
    SYNTAX_ERROR = 1
    TIMED_OUT = 2
    OTHER_ERROR = 3
    TOO_LARGE = 4
```

**Recovery Strategies:**
- Syntax Error → Controller revises query
- Timeout → Controller simplifies query or adds LIMIT
- Too Large → Controller adds constraints or LIMIT
- No Results → Controller explores alternative properties

### Prompt Templates

**Controller Prompt** (`controller.prompt`)
```
You are an expert in writing SPARQL queries for Wikidata. Your task is to
construct a SPARQL query that answers the user's question.

Available actions:
- execute_sparql(SPARQL): Test a query
- get_wikidata_entry(QID): Explore an entity
- search_wikidata(string): Find entities/properties
- get_property_examples(PID): See property usage
- stop(): Complete the task

Strategy:
1. Start by constructing very simple fragments
2. Execute each fragment to verify correctness
3. Confirm assumptions about Wikidata structure
4. Gradually build complete query
5. Do NOT repeat same action
```

**Format Actions Prompt** (`format_actions.prompt`)
```
Format your response as JSON:
{
    'thought': 'your reasoning',
    'action_name': 'execute_sparql',
    'action_argument': 'SELECT ...'
}
```

**Prune Edges Prompt** (`prune_outgoing_edges.prompt`)
```
Given entity properties and the question, filter to only relevant properties.

Example:
Question: "Where did Barack Obama sr. attend school?"
Entity: Barack_Obama_sr (Q649593)
Properties: [50+ properties]

Relevant properties:
- educated_at (P69)
- place_of_birth (P19)
- country_of_citizenship (P27)
```

### Wikidata API Functions

**execute_sparql(sparql: str)**
- Endpoint: https://query.wikidata.org/sparql
- Method: POST
- Headers: Accept: application/sparql-results+json
- Timeout: 30 seconds
- Returns: JSON results or error

**get_outgoing_edges(qid: str)**
- Uses: Wikidata API special page
- URL: https://www.wikidata.org/wiki/Special:EntityData/{qid}.json
- Returns: All properties and qualifiers for entity
- Parsed: Property-value pairs with labels

**search_span(span: str, limit: int = 10)**
- Uses: Wikidata search API
- URL: https://www.wikidata.org/w/api.php?action=wbsearchentities
- Parameters: search={span}, limit={limit}
- Returns: List of matching entities with QID and description

**get_property_examples(pid: str, limit: int = 10)**
- Executes: SELECT ?subject ?object WHERE { ?subject wdt:{pid} ?object } LIMIT {limit}
- Returns: Sample usage of the property
- Includes: Labels for both subjects and objects

### Deduplication Strategy

**Action Deduplication:**
```python
# Track all actions in state
state.actions.append(f"{action_name}({action_argument})")

# Before executing new action
if new_action in state.actions:
    # Don't repeat
    controller_decides_different_action()
```

**Benefits:**
- Prevents infinite loops
- Avoids wasted iterations
- Forces exploration of new approaches

### Query Optimization Techniques

**1. Inverse Properties**
```
# More efficient for hierarchical queries
BEFORE: ?item wdt:P31/wdt:P279* wd:Q5
AFTER:  wd:Q5 ^wdt:P279*/^wdt:P31 ?item
```

**2. COUNT Optimization**
```
# Faster counting
BEFORE: COUNT(?item)
AFTER:  COUNT(*)
```

**3. Service Label Optimization**
```
# Single pattern for all labels
SERVICE wikibase:label {
  bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
}
# Automatically provides ?xLabel, ?yLabel, etc.
```

**4. Property Path Simplification**
```
# Use direct inverse when possible
BEFORE: ?x wdt:P31 ?type . ?type wdt:P279 ?parent
AFTER:  ?x wdt:P31/wdt:P279 ?parent
```

### Configuration Parameters

**Max Iterations:** 15
- Prevents infinite loops
- Forces efficient exploration

**SPARQL Timeout:** 30 seconds
- Balances thoroughness with responsiveness
- Prevents hanging on complex queries

**Search Results Limit:** 10
- Provides enough options for disambiguation
- Avoids overwhelming with choices

**Property Examples Limit:** 10
- Sufficient to understand usage patterns
- Fast to retrieve and process

**Result Size Limit:** 10MB
- Prevents memory issues
- Signals overly broad queries

### File Structure

```
spinach-claudetest/
├── spinach_agent/
│   ├── part_to_whole_parser.py      # Main state machine
│   ├── parser_state.py               # State and SparqlQuery classes
│   ├── parser_utils.py               # LangChain utility functions
│   └── prompts/
│       ├── controller.prompt         # Main decision-making prompt
│       ├── format_actions.prompt     # JSON formatting prompt
│       ├── prune_outgoing_edges.prompt  # Property filtering
│       └── extract_final_sparql.prompt  # Last-resort extraction
├── wikidata_utils.py                 # SPARQL execution & Wikidata API
└── datasets/
    └── qald_9_plus/en/
        └── spinach_output_test.json  # Benchmark results
```

### Example Trace

**Question:** "What is the time zone of Salt Lake City?"

**Turn 1:**
```
Action: search_wikidata("Salt Lake City")
Result: [
  {id: "Q23337", label: "Salt Lake City", description: "capital city of Utah"},
  {id: "Q108128", label: "Salt Lake City", description: "1984 film"},
  ...
]
Thought: Q23337 is the city in Utah
```

**Turn 2:**
```
Action: get_wikidata_entry("Q23337")
Result: {
  "instance_of": ["city", "big_city"],
  "country": ["United States"],
  "located_in": ["Utah"],
  "time_zone": ["Mountain Time Zone", "UTC−07:00"],
  ...
}
Thought: P421 (time_zone) has the answer directly
```

**Turn 3:**
```
Action: execute_sparql("
  SELECT DISTINCT ?timeZone ?timeZoneLabel WHERE {
    wd:Q23337 wdt:P421 ?timeZone.
    SERVICE wikibase:label {
      bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'.
    }
  }
")
Result: {
  "results": [
    {"timeZone": "wd:Q2134", "timeZoneLabel": "Mountain Time Zone"},
    {"timeZone": "wd:Q7041", "timeZoneLabel": "UTC−07:00"}
  ],
  "status": "OK"
}
Thought: Query successful, have answer
```

**Turn 4:**
```
Action: stop()
Final Answer: The time zone of Salt Lake City is Mountain Time Zone (UTC−07:00)
```

### Performance Metrics

Based on QALD-9-Plus benchmark (`spinach_output_test.json`):

- **Average iterations:** 4-6 steps per question
- **Success rate:** High accuracy on well-formed questions
- **Common failure modes:**
  - Ambiguous entity references
  - Complex temporal reasoning
  - Multi-hop property chains with qualifiers

### Integration with LangChain

**LLM Integration:**
```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4", temperature=0)
controller_chain = prompt | llm | output_parser
```

**Chains Used:**
- `sparql_string_to_sparql_object`: Convert string to SparqlQuery
- `execute_sparql_object`: Execute and return results
- `get_prune_edges_chain`: Filter irrelevant properties

### Extension Points

The system can be extended by:

1. **Adding new actions** to the state machine
2. **Customizing prompts** for different domains
3. **Adjusting parameters** (max iterations, timeouts)
4. **Adding preprocessing** for question understanding
5. **Implementing caching** for common entity/property lookups

---

## Appendix B: SPARQL Query Examples

### Basic Entity Property Query
```sparql
# What is Frank Herbert's profession?
SELECT DISTINCT ?profession ?professionLabel WHERE {
  wd:Q7934 wdt:P106 ?profession.
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
  }
}
```

### Shared Property Value Query
```sparql
# Which artists were born on the same date as Rachel Stevens (1978-04-09)?
SELECT DISTINCT ?artist ?artistLabel WHERE {
  ?artist wdt:P569 '1978-04-09T00:00:00Z'^^xsd:dateTime.
  ?artist wdt:P106 wd:Q483501.
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
  }
}
```

### Property Path Query
```sparql
# What are the subclasses of mammals?
SELECT DISTINCT ?subclass ?subclassLabel WHERE {
  ?subclass wdt:P279+ wd:Q7377.
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
  }
}
```

### Temporal Filter Query
```sparql
# Which US presidents were in office during the Vietnam War (1955-1975)?
SELECT DISTINCT ?president ?presidentLabel WHERE {
  VALUES ?president { wd:Q9588 wd:Q9582 wd:Q9640 wd:Q9696 wd:Q9916 }
  ?president wdt:P39 wd:Q11696.
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
  }
}
```

### Multi-Constraint Query
```sparql
# Paintings by Leonardo da Vinci in French museums
SELECT DISTINCT ?painting ?paintingLabel ?museum ?museumLabel WHERE {
  ?painting wdt:P170 wd:Q762.           # creator: Leonardo da Vinci
  ?painting wdt:P31 wd:Q3305213.        # instance of: painting
  ?painting wdt:P276 ?museum.           # location: museum
  ?museum wdt:P17 wd:Q142.              # country: France
  ?museum wdt:P31/wdt:P279* wd:Q33506.  # instance of: museum (or subclass)
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
  }
}
```

### Aggregation Query
```sparql
# Count of movies directed by each director born in Paris
SELECT ?director ?directorLabel (COUNT(?movie) AS ?movieCount) WHERE {
  ?director wdt:P19 wd:Q90.          # place of birth: Paris
  ?director wdt:P106 wd:Q2526255.    # occupation: film director
  ?movie wdt:P57 ?director.          # director: ?director
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
  }
}
GROUP BY ?director ?directorLabel
ORDER BY DESC(?movieCount)
LIMIT 10
```

---

## Appendix C: Common Wikidata Properties

Understanding frequently-used properties accelerates query composition:

### People Properties
- **P31**: instance of
- **P569**: date of birth
- **P570**: date of death
- **P19**: place of birth
- **P20**: place of death
- **P106**: occupation
- **P27**: country of citizenship
- **P735**: given name
- **P734**: family name
- **P26**: spouse
- **P40**: child
- **P22**: father
- **P25**: mother

### Organization Properties
- **P31**: instance of
- **P571**: inception date
- **P576**: dissolved/abolished
- **P159**: headquarters location
- **P112**: founder
- **P1128**: employees
- **P452**: industry

### Location Properties
- **P31**: instance of
- **P17**: country
- **P131**: located in administrative territory
- **P625**: coordinate location
- **P421**: time zone
- **P1082**: population
- **P2046**: area

### Creative Work Properties
- **P31**: instance of
- **P50**: author
- **P170**: creator
- **P577**: publication date
- **P123**: publisher
- **P407**: language of work
- **P136**: genre
- **P921**: main subject

### Relationship Properties
- **P279**: subclass of
- **P361**: part of
- **P527**: has part
- **P1376**: capital of
- **P47**: shares border with

---

## Appendix D: Troubleshooting Guide

### Problem: Empty Results

**Diagnostic Steps:**
1. Test each constraint independently
2. Verify entity/property IDs are correct
3. Check value format (especially dates, numbers)
4. Explore example uses of properties

**Example:**
```
Query returns nothing:
SELECT ?x WHERE { wd:Q762 wdt:P800 ?x. ?x wdt:P31 wd:Q3305213. }

Test component 1:
SELECT ?x WHERE { wd:Q762 wdt:P800 ?x. }
→ Returns works by Leonardo da Vinci ✓

Test component 2:
SELECT ?type WHERE { wd:Q175936 wdt:P31 ?type. }  # Test with known painting
→ Returns Q3305213 ✓

Diagnose: Components work separately. Possible issue: Not all notable works are paintings.
Solution: Query is correct; results may genuinely be empty or you need alternative path.
```

### Problem: Query Timeout

**Diagnostic Steps:**
1. Add LIMIT clause (e.g., LIMIT 100)
2. Make query more selective (add constraints)
3. Avoid unbounded property paths
4. Consider inverse property formulation

**Example:**
```
Timeout:
SELECT ?person WHERE { ?person wdt:P31/wdt:P279* wd:Q5. }

Fix 1 - Add LIMIT:
SELECT ?person WHERE { ?person wdt:P31/wdt:P279* wd:Q5. } LIMIT 1000

Fix 2 - More specific:
SELECT ?person WHERE {
  ?person wdt:P31 wd:Q5.           # Direct instance of human
  ?person wdt:P27 wd:Q30.          # Citizen of USA
} LIMIT 1000

Fix 3 - Inverse:
SELECT ?person WHERE { wd:Q5 ^wdt:P279*/^wdt:P31 ?person. } LIMIT 1000
```

### Problem: Syntax Error

**Common Causes:**
1. Missing period at end of triple
2. Incorrect prefix (use `wd:` and `wdt:`)
3. Unclosed brackets or braces
4. Invalid property path syntax

**Example:**
```
Error:
SELECT ?x WHERE { wd:Q762 P800 ?x }

Fix:
SELECT ?x WHERE { wd:Q762 wdt:P800 ?x. }
         Added prefix ↑      Added period ↑
```

### Problem: Wrong Results

**Diagnostic Steps:**
1. Verify entity disambiguation (correct Q-number)
2. Check property semantics with examples
3. Examine individual result entities
4. Consider need for additional filters

**Example:**
```
Question: "Paintings by Leonardo da Vinci"
Query returns sculptures too.

Diagnose: Check what P800 (notable works) returns
ACTION: get_property_examples(P800)
RESULT: Returns all creative works, not just paintings

Fix: Add constraint on instance type
SELECT ?x WHERE {
  wd:Q762 wdt:P800 ?x.
  ?x wdt:P31 wd:Q3305213.  # instance of painting
}
```
