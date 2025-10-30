"""Sample ontologies for testing."""

from baml_client.types import EntityType, RelationType, Ontology


def create_simple_ontology():
    """Create a simple ontology for basic tests."""
    entity_types = [
        EntityType(name="Person", description="A human being"),
        EntityType(name="Organization", description="A company or institution")
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
        EntityType(name="Person", description="A human being"),
        EntityType(name="Organization", description="A company or institution"),
        EntityType(name="Location", description="A geographic place"),
        EntityType(name="Technology", description="A programming language or framework"),
        EntityType(name="Product", description="A product or service")
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
        {"name": "Person", "description": "A human being"},
        {"name": "Organization", "description": "A company or institution"}
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
        {"name": "Person", "description": "A human being"},
        {"name": "Organization", "description": "A company or institution"},
        {"name": "Location", "description": "A geographic place"},
        {"name": "Technology", "description": "A programming language or framework"}
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

