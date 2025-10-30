"""Sample texts for testing."""


SIMPLE_TEXT = """
Alice Johnson works at TechCorp. TechCorp is located in San Francisco.
"""

COMPLEX_TEXT = """
Alice Johnson is a senior software engineer who works at TechCorp, a rapidly 
growing technology company. TechCorp is headquartered in San Francisco, where 
it occupies a modern office building. The company was founded in 2015 by Bob Smith.

Alice has been with TechCorp since its early days. She is particularly skilled in Python 
development and frequently uses React to build user interfaces.
"""

TEXT_WITH_NEWLINES = """
John Smith
works at
Google Inc.
in Mountain View,
California.
"""

TEXT_WITH_EXTRA_WHITESPACE = """
Mary   Wilson    works    at    
    DataCorp    which    is    located    in    
New York City.
"""

MEDICAL_TEXT = """
Dr. Sarah Chen led a clinical trial at Massachusetts General Hospital 
to evaluate Medication A for treating chronic migraines. The trial showed 
promising results with 75% of patients experiencing symptom reduction.
"""

BUSINESS_TEXT = """
Acme Corporation announced a strategic investment in StartupXYZ. The investment, 
led by CEO Jennifer Adams, totals $50 million and will help StartupXYZ expand 
its artificial intelligence platform. Acme Corporation is based in Boston.
"""

LEGAL_TEXT = """
In the case Smith v. Johnson, the plaintiff filed a lawsuit in the Superior Court 
of California. The defendant's attorney, Lisa Brown of Brown & Associates Law Firm, 
filed a motion to dismiss.
"""

# Text for entity consistency testing
TEXT_FIRST_MENTION = """
Alice works at TechCorp.
"""

TEXT_SECOND_MENTION = """
Alice Johnson is now using TypeScript. TechCorp is expanding to New York.
"""

# Texts with difficult character span matching
TEXT_WITH_QUOTES = """
Alice Johnson stated, "I love working at TechCorp," during the company meeting.
"""

TEXT_WITH_PUNCTUATION = """
TechCorp (founded in 2015) is located in San Francisco, California, U.S.A.
"""

# Short texts for minimal scope testing
MINIMAL_SCOPE_TEXT = """
John works at Google.
"""

# Medium texts for balanced scope testing
BALANCED_SCOPE_TEXT = """
Sarah Chen is a data scientist at DataCorp, a technology company based in Seattle.
She specializes in machine learning and uses Python extensively in her work.
DataCorp was founded by Michael Lee in 2018 and has grown to 200 employees.
"""

# Long texts for comprehensive scope testing
COMPREHENSIVE_SCOPE_TEXT = """
TechInnovate is a multinational technology corporation headquartered in Silicon Valley, 
California. The company was founded in 2010 by Dr. Emily Watson, a computer scientist 
with expertise in artificial intelligence and distributed systems.

The company's flagship product, CloudOS, is a cloud-based operating system that serves 
over 10 million users worldwide. CloudOS was developed by a team led by Chief Technology 
Officer David Kim, who joined TechInnovate in 2012.

TechInnovate has offices in five countries: United States, United Kingdom, Germany, 
Japan, and Singapore. The London office, which opened in 2015, focuses on financial 
technology partnerships. The Berlin office specializes in automotive technology, 
partnering with major German car manufacturers.

The company has made several strategic acquisitions. In 2018, TechInnovate acquired 
AIStartup for $200 million to enhance its machine learning capabilities. The acquisition 
was led by Vice President of Corporate Development, Lisa Anderson.

Dr. Watson continues to serve as CEO and remains actively involved in research and 
development. She holds 15 patents in distributed computing and was named one of 
Time Magazine's 100 Most Influential People in 2020.

The company recently announced a partnership with QuantumCorp to develop quantum 
computing solutions. This partnership, announced by Chief Strategy Officer Robert Chen, 
represents TechInnovate's entry into the quantum computing market.
"""

