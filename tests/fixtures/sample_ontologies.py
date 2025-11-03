"""Sample ontologies for testing."""

from spindle.baml_client.types import EntityType, RelationType, Ontology, AttributeDefinition


def create_simple_ontology():
    """Create a simple ontology for basic tests."""
    entity_types = [
        EntityType(
            name="Person",
            description="A human being",
            attributes=[]
        ),
        EntityType(
            name="Organization",
            description="A company or institution",
            attributes=[]
        )
    ]
    
    relation_types = [
        RelationType(
            name="works_at",
            description="Employment relationship",
            domain="Person",
            range="Organization"
        )
    ]
    
    return Ontology(entity_types=entity_types, relation_types=relation_types)


def create_complex_ontology():
    """Create a more complex ontology for comprehensive tests."""
    entity_types = [
        EntityType(
            name="Person",
            description="A human being",
            attributes=[
                AttributeDefinition(
                    name="birth_date",
                    type="date",
                    description="Date of birth"
                ),
                AttributeDefinition(
                    name="nationality",
                    type="string",
                    description="Country of citizenship"
                )
            ]
        ),
        EntityType(
            name="Organization",
            description="A company or institution",
            attributes=[
                AttributeDefinition(
                    name="founded_date",
                    type="date",
                    description="Date the organization was founded"
                ),
                AttributeDefinition(
                    name="employee_count",
                    type="int",
                    description="Number of employees"
                )
            ]
        ),
        EntityType(
            name="Location",
            description="A geographic place",
            attributes=[]
        ),
        EntityType(
            name="Technology",
            description="A programming language or framework",
            attributes=[
                AttributeDefinition(
                    name="release_year",
                    type="int",
                    description="Year the technology was first released"
                )
            ]
        ),
        EntityType(
            name="Product",
            description="A product or service",
            attributes=[
                AttributeDefinition(
                    name="price",
                    type="float",
                    description="Price in USD"
                )
            ]
        )
    ]
    
    relation_types = [
        RelationType(
            name="works_at",
            description="Employment relationship",
            domain="Person",
            range="Organization"
        ),
        RelationType(
            name="located_in",
            description="Physical location relationship",
            domain="Organization",
            range="Location"
        ),
        RelationType(
            name="uses",
            description="Technology usage relationship",
            domain="Person",
            range="Technology"
        ),
        RelationType(
            name="founded",
            description="Founding relationship",
            domain="Person",
            range="Organization"
        ),
        RelationType(
            name="develops",
            description="Product development relationship",
            domain="Organization",
            range="Product"
        )
    ]
    
    return Ontology(entity_types=entity_types, relation_types=relation_types)


def get_simple_entity_types_dict():
    """Get simple entity types as dict format (for create_ontology function)."""
    return [
        {"name": "Person", "description": "A human being", "attributes": []},
        {"name": "Organization", "description": "A company or institution", "attributes": []}
    ]


def get_simple_relation_types_dict():
    """Get simple relation types as dict format (for create_ontology function)."""
    return [
        {
            "name": "works_at",
            "description": "Employment relationship",
            "domain": "Person",
            "range": "Organization"
        }
    ]


def get_complex_entity_types_dict():
    """Get complex entity types as dict format."""
    return [
        {
            "name": "Person",
            "description": "A human being",
            "attributes": [
                {"name": "birth_date", "type": "date", "description": "Date of birth"},
                {"name": "nationality", "type": "string", "description": "Country of citizenship"}
            ]
        },
        {
            "name": "Organization",
            "description": "A company or institution",
            "attributes": [
                {"name": "founded_date", "type": "date", "description": "Date the organization was founded"},
                {"name": "employee_count", "type": "int", "description": "Number of employees"}
            ]
        },
        {
            "name": "Location",
            "description": "A geographic place",
            "attributes": []
        },
        {
            "name": "Technology",
            "description": "A programming language or framework",
            "attributes": [
                {"name": "release_year", "type": "int", "description": "Year the technology was first released"}
            ]
        }
    ]


def get_complex_relation_types_dict():
    """Get complex relation types as dict format."""
    return [
        {
            "name": "works_at",
            "description": "Employment relationship",
            "domain": "Person",
            "range": "Organization"
        },
        {
            "name": "located_in",
            "description": "Physical location relationship",
            "domain": "Organization",
            "range": "Location"
        },
        {
            "name": "uses",
            "description": "Technology usage relationship",
            "domain": "Person",
            "range": "Technology"
        }
    ]


def create_campaign_ontology():
    """Create an ontology for healthcare campaign examples."""
    entity_types = [
        EntityType(
            name="Campaign",
            description="A marketing or outreach campaign",
            attributes=[
                AttributeDefinition(
                    name="campaign_launch_dt",
                    type="date",
                    description="The date the campaign was launched"
                ),
                AttributeDefinition(
                    name="campaign_completion_dt",
                    type="date",
                    description="The date the campaign completed or is planned to complete"
                ),
                AttributeDefinition(
                    name="campaign_type",
                    type="string",
                    description="Type of campaign (e.g., next-best-action, preventive care)"
                )
            ]
        ),
        EntityType(
            name="PatientSegment",
            description="A group or segment of patients",
            attributes=[
                AttributeDefinition(
                    name="segment_size",
                    type="int",
                    description="Number of patients in this segment"
                )
            ]
        ),
        EntityType(
            name="HealthAction",
            description="A recommended health action or intervention",
            attributes=[]
        )
    ]
    
    relation_types = [
        RelationType(
            name="targets",
            description="Campaign targets a patient segment",
            domain="Campaign",
            range="PatientSegment"
        ),
        RelationType(
            name="recommends",
            description="Campaign recommends a health action",
            domain="Campaign",
            range="HealthAction"
        )
    ]
    
    return Ontology(entity_types=entity_types, relation_types=relation_types)

