"""Ontology recommendation and extension endpoints."""

from fastapi import APIRouter, HTTPException, status

from spindle.api.models import (
    OntologyExtensionAnalysisRequest,
    OntologyExtensionAnalysisResponse,
    OntologyExtensionApplyRequest,
    OntologyExtensionApplyResponse,
    OntologyRecommendationRequest,
    OntologyRecommendationResponse,
    RecommendAndExtractRequest,
    RecommendAndExtractResponse,
)
from spindle.api.utils import (
    convert_baml_to_dict,
    serialize_extraction_result,
    serialize_ontology,
)
from spindle.baml_client.types import Ontology, OntologyExtension, Triple
from spindle.extraction.recommender import OntologyRecommender

router = APIRouter()


def _ontology_from_dict(ontology_dict: dict) -> Ontology:
    """Convert dictionary to Ontology object."""
    try:
        return Ontology.model_validate(ontology_dict)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ontology format: {str(e)}",
        )


def _extension_from_dict(extension_dict: dict) -> OntologyExtension:
    """Convert dictionary to OntologyExtension object."""
    try:
        return OntologyExtension.model_validate(extension_dict)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid extension format: {str(e)}",
        )


def _triples_from_dicts(triple_dicts: list) -> list:
    """Convert list of dictionaries to Triple objects."""
    try:
        return [Triple.model_validate(t) for t in triple_dicts]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid triple format: {str(e)}",
        )


# ============================================================================
# Ontology Recommendation
# ============================================================================


@router.post("/recommend", response_model=OntologyRecommendationResponse)
async def recommend_ontology(request: OntologyRecommendationRequest):
    """Recommend an ontology based on text analysis.
    
    This endpoint analyzes text and recommends an appropriate ontology
    for knowledge graph extraction.
    
    Args:
        request: Recommendation parameters
        
    Returns:
        Recommended ontology with reasoning
        
    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors
    """
    try:
        # Create recommender
        recommender = OntologyRecommender()
        
        # Recommend ontology
        result = recommender.recommend(
            text=request.text,
            scope=request.scope.value,
        )
        
        # Serialize ontology
        ontology_dict = serialize_ontology(result.ontology)
        
        return OntologyRecommendationResponse(
            ontology=ontology_dict,
            text_purpose=result.text_purpose,
            reasoning=result.reasoning,
            entity_type_count=len(result.ontology.entity_types),
            relation_type_count=len(result.ontology.relation_types),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ontology recommendation failed: {str(e)}",
        )


# ============================================================================
# Ontology Extension
# ============================================================================


@router.post("/extend/analyze", response_model=OntologyExtensionAnalysisResponse)
async def analyze_ontology_extension(request: OntologyExtensionAnalysisRequest):
    """Analyze whether an ontology needs extension for new text.
    
    This endpoint conservatively analyzes if the current ontology is
    sufficient or needs extension to handle new text.
    
    Args:
        request: Extension analysis parameters
        
    Returns:
        Extension analysis result
        
    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors
    """
    try:
        # Parse ontology
        current_ontology = _ontology_from_dict(request.current_ontology)
        
        # Create recommender
        recommender = OntologyRecommender()
        
        # Analyze extension needs
        extension = recommender.analyze_extension(
            text=request.text,
            current_ontology=current_ontology,
            scope=request.scope.value,
        )
        
        # Serialize result
        extension_dict = convert_baml_to_dict(extension)
        
        return OntologyExtensionAnalysisResponse(
            needs_extension=extension.needs_extension,
            new_entity_types=extension_dict.get("new_entity_types", []),
            new_relation_types=extension_dict.get("new_relation_types", []),
            critical_information_at_risk=extension.critical_information_at_risk,
            reasoning=extension.reasoning,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extension analysis failed: {str(e)}",
        )


@router.post("/extend/apply", response_model=OntologyExtensionApplyResponse)
async def apply_ontology_extension(request: OntologyExtensionApplyRequest):
    """Apply an ontology extension to create an extended ontology.
    
    This endpoint applies the extension from analyze_extension to create
    a new ontology with the additional types.
    
    Args:
        request: Extension application parameters
        
    Returns:
        Extended ontology
        
    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors
    """
    try:
        # Parse ontology and extension
        current_ontology = _ontology_from_dict(request.current_ontology)
        extension = _extension_from_dict(request.extension)
        
        # Create recommender
        recommender = OntologyRecommender()
        
        # Apply extension
        extended_ontology = recommender.extend_ontology(
            current_ontology=current_ontology,
            extension=extension,
        )
        
        # Serialize result
        ontology_dict = serialize_ontology(extended_ontology)
        
        return OntologyExtensionApplyResponse(
            extended_ontology=ontology_dict,
            entity_type_count=len(extended_ontology.entity_types),
            relation_type_count=len(extended_ontology.relation_types),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extension application failed: {str(e)}",
        )


# ============================================================================
# Combined Operations
# ============================================================================


@router.post("/recommend-and-extract", response_model=RecommendAndExtractResponse)
async def recommend_and_extract(request: RecommendAndExtractRequest):
    """Recommend ontology and extract triples in one operation.
    
    This endpoint combines ontology recommendation and triple extraction
    for convenience.
    
    Args:
        request: Combined operation parameters
        
    Returns:
        Recommended ontology and extracted triples
        
    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors
    """
    try:
        # Parse existing triples if provided
        existing_triples = None
        if request.existing_triples:
            existing_triples = _triples_from_dicts(request.existing_triples)
        
        # Create recommender
        recommender = OntologyRecommender()
        
        # Recommend and extract
        recommendation, extraction = recommender.recommend_and_extract(
            text=request.text,
            source_name=request.source_name,
            source_url=request.source_url,
            scope=request.scope.value,
            existing_triples=existing_triples,
        )
        
        # Serialize results
        ontology_dict = serialize_ontology(recommendation.ontology)
        extraction_dict = serialize_extraction_result(extraction)
        
        return RecommendAndExtractResponse(
            ontology=ontology_dict,
            text_purpose=recommendation.text_purpose,
            ontology_reasoning=recommendation.reasoning,
            triples=extraction_dict["triples"],
            extraction_reasoning=extraction_dict["reasoning"],
            triple_count=len(extraction_dict["triples"]),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommend and extract failed: {str(e)}",
        )

