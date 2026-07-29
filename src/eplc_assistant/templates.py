"""Supported UI templates mapped to their persisted vector-index metadata."""

PHASE_TEMPLATE_SOURCES: dict[str, dict[str, tuple[str, ...]]] = {
    "Requirement": {
        "Physical Data Model": ("Physical Data Model_embedding.json",),
    },
    "Design": {
        "Capacity Planning": ("CDC_UP_Capacity_Planning_embedding (2).json",),
        "Contingency Planning": (
            "CDC_UP_Contingency_Planning_Final_Embedding.json",
            "CDC_UP_Contingency_Planning_embedding.json",
        ),
        "Data Conversion Plan": ("EPLC_Data_Conversion_Plan_embedding.json",),
        "Implementation Plan": ("EPLC_Implementation_Plan_embedding.json",),
        "Interface Control": ("CDC_UP_Interface_Control_embedding (1).json",),
    },
    "Development": {
        "Operation and Maintenance Manual": (
            "CDC_UP_Operation_Maintenance_Manual_Template",
        ),
        "Test Case": ("CDC_UP_Test_Case_Template (Cleaned)",),
        "Training Plan": ("Training Plan Template (v1.0)",),
    },
    "Implementation": {
        "Acquisition Strategy": (
            "EPLC_Acquisition_Strategy_Template_embedding.json",
        ),
        "Business Impact Analysis": ("Business Analysis Impact_embedding.json",),
        "Lessons Learned Log": ("Lessons_Learned_Log_embedding.json",),
        "Lessons Learned Post-Project Survey": (
            "CDC_UP_Lessons_Learned_Post_Project_Survey_embedding_output.json",
        ),
        "Service Level Agreement / MOU": ("SLA_MOU_embedding.json",),
        "System of Records Notice": (
            "System_of_Records_Notice_embedding.json",
        ),
    },
}

PHASE_TEMPLATES = {
    phase: tuple(templates)
    for phase, templates in PHASE_TEMPLATE_SOURCES.items()
}
