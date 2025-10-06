#!/usr/bin/env python3
"""
FEMM Implementation Planner Crew - Phase 2 Implementation

This module implements the FEMM planning system that:
1. Analyzes selected research paper in detail
2. Designs FEMM geometry, materials, and boundary conditions
3. Plans ML architecture and data pipeline
4. Creates step-by-step execution plan

Usage:
    from femm_planner_crew import create_femm_planning_crew
    crew = create_femm_planning_crew()
    result = crew.kickoff(inputs={"research_id": 3, "research_context": context_json})
"""

import os
import json
import ast
import re
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_llm_client():
    """Initialize and return the LLM client"""
    return LLM(
        model="ollama/qwen3:8b",
        base_url="http://localhost:11434"
    )

class FEMMValidatorTool(BaseTool):
    """Tool to validate FEMM API calls and geometry"""
    name: str = "FEMM Validator"
    description: str = "Validate FEMM API calls and geometry specifications"

    def __init__(self):
        super().__init__()

    def _run(self, femm_code: str) -> str:
        """Validate FEMM code for API correctness"""
        try:
            # Check for valid FEMM function calls
            femm_functions = [
                'mi_addnode', 'mi_addsegment', 'mi_addblocklabel', 'mi_addarc',
                'mi_drawline', 'mi_drawrectangle', 'mi_drawarc',
                'mi_setnodeprop', 'mi_setblockprop', 'mi_setsegmentprop',
                'mi_probdef', 'mi_analyze', 'mi_loadsolution',
                'mi_createmesh', 'mi_showmesh', 'mi_purgemesh',
                'mi_getmaterial', 'mi_addmaterial', 'mi_addboundprop',
                'mo_getpointvalues', 'mo_lineintegral', 'mo_blockintegral'
            ]

            found_functions = []
            for func in femm_functions:
                if func in femm_code:
                    found_functions.append(func)

            # Check for proper FEMM workflow
            workflow_checks = {
                'geometry_created': any(func in femm_code for func in ['mi_addnode', 'mi_drawline', 'mi_drawrectangle']),
                'materials_defined': 'mi_addmaterial' in femm_code or 'mi_getmaterial' in femm_code,
                'boundary_conditions': 'mi_addboundprop' in femm_code,
                'analysis_run': 'mi_analyze' in femm_code,
                'results_extracted': any(func in femm_code for func in ['mo_getpointvalues', 'mo_lineintegral'])
            }

            return json.dumps({
                'valid_functions': found_functions,
                'workflow_checks': workflow_checks,
                'is_valid': len(found_functions) > 0
            })

        except Exception as e:
            return f"Error validating FEMM code: {str(e)}"

class LibraryValidatorTool(BaseTool):
    """Tool to validate Python library availability"""
    name: str = "Library Validator"
    description: str = "Check if required Python libraries are available"

    def __init__(self):
        super().__init__()

    def _run(self, libraries: List[str]) -> str:
        """Check library availability"""
        try:
            available = []
            unavailable = []

            for lib in libraries:
                try:
                    # Try to import the library
                    __import__(lib.replace('-', '_'))
                    available.append(lib)
                except ImportError:
                    unavailable.append(lib)

            return json.dumps({
                'available': available,
                'unavailable': unavailable,
                'all_available': len(unavailable) == 0
            })

        except Exception as e:
            return f"Error checking libraries: {str(e)}"

class ResearchContext(BaseModel):
    """Schema for research context input"""
    research_id: int
    title: str
    authors: List[str]
    summary: str
    methodology: str
    femm_relevance: float
    complexity: str
    source_url: str
    key_contributions: List[str]

class ImplementationPlan(BaseModel):
    """Schema for implementation plan output"""
    research_title: str
    implementation_plan: Dict

def create_femm_planning_crew():
    """Create the FEMM planning crew with 4 specialized agents"""

    # Initialize tools
    femm_validator = FEMMValidatorTool()
    library_validator = LibraryValidatorTool()

    # Get LLM client
    llm = get_llm_client()

    # Agent 1: Paper Deep Analyzer
    paper_analyzer = Agent(
        role="Research Paper Deep Analyst",
        goal="Perform in-depth analysis of selected research paper for FEMM implementation",
        backstory="""You are an expert at analyzing academic papers in electromagnetic design
        and machine learning. You can extract detailed methodology, equations, algorithms,
        and identify specific components that can be implemented in FEMM.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=3,  # Guardrail: limit iterations
    )

    # Agent 2: FEMM Geometry Planner
    femm_geometry_planner = Agent(
        role="Electromagnetic Design Specialist",
        goal="Design FEMM geometry, materials, and boundary conditions based on research paper",
        backstory="""You are a specialist in finite element electromagnetic simulation using FEMM.
        You understand how to translate research concepts into practical FEMM geometry,
        material properties, and boundary conditions. You reference real FEMM implementations.""",
        verbose=True,
        allow_delegation=False,
        tools=[femm_validator],
        llm=llm,
        max_iter=4,  # Guardrail: limit iterations
    )

    # Agent 3: ML Architecture Designer
    ml_architect = Agent(
        role="Machine Learning Engineer",
        goal="Design neural network architecture and data pipeline for the research implementation",
        backstory="""You are an expert in machine learning systems, particularly neural networks
        for engineering applications. You design efficient architectures that integrate well
        with FEMM simulations and understand the computational constraints.""",
        verbose=True,
        allow_delegation=False,
        tools=[library_validator],
        llm=llm,
        max_iter=3,  # Guardrail: limit iterations
    )

    # Agent 4: Integration Architect
    integration_architect = Agent(
        role="Systems Integration Specialist",
        goal="Create comprehensive execution plan connecting FEMM simulation with ML training",
        backstory="""You are an expert at integrating complex systems, particularly FEMM-based
        electromagnetic simulations with machine learning pipelines. You create robust,
        executable workflows that handle the unique challenges of FEMM+ML integration.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=2,  # Guardrail: limit iterations
    )

    # Task 1: Deep paper analysis
    paper_analysis_task = Task(
        description="""
        Perform in-depth analysis of the selected research paper.

        Research Context: {research_context}

        Analyze:
        1. Core methodology and algorithms
        2. Mathematical equations and formulations
        3. Experimental setup and parameters
        4. Key findings and contributions
        5. Specific components that can be implemented in FEMM
        6. Data requirements for ML components

        Extract specific details that will inform FEMM geometry and ML architecture design.
        """,
        expected_output="Detailed analysis of paper methodology and technical components",
        agent=paper_analyzer,
    )

    # Task 2: FEMM geometry and setup planning
    femm_planning_task = Task(
        description="""
        Design FEMM implementation based on paper analysis.

        Paper Analysis: {paper_analysis}

        Design:
        1. Problem type (magnetostatic, electrostatic, heat flow, current flow)
        2. Geometry specifications (dimensions, components, design space)
        3. Material properties (magnetic, electrical, thermal)
        4. Boundary conditions (Dirichlet, Neumann, periodic)
        5. Circuit definitions (coils, currents, voltages)
        6. Mesh strategy (element size, refinement)
        7. Analysis parameters (frequency, precision, depth)

        Reference your existing Chp4/Chp5 FEMM implementations for similar electromagnetic actuators.
        Ensure all FEMM API calls are valid and follow proper workflow.
        """,
        expected_output="Complete FEMM setup specifications with geometry, materials, and analysis parameters",
        agent=femm_geometry_planner,
        context=[paper_analysis_task],
    )

    # Task 3: ML architecture design
    ml_design_task = Task(
        description="""
        Design ML architecture and data pipeline for the research implementation.

        FEMM Setup: {femm_setup}

        Design:
        1. Neural network architecture (CNN, RNN, Transformer, etc.)
        2. Input/output specifications (data shapes, features)
        3. Training strategy (supervised, unsupervised, reinforcement)
        4. Data generation pipeline (FEMM simulation → dataset)
        5. Preprocessing requirements (normalization, augmentation)
        6. Model evaluation metrics
        7. Required ML libraries and frameworks

        Consider computational constraints and FEMM integration requirements.
        """,
        expected_output="Complete ML architecture design with data pipeline specifications",
        agent=ml_architect,
        context=[femm_planning_task],
    )

    # Task 4: Integration and execution planning
    integration_task = Task(
        description="""
        Create comprehensive execution plan integrating FEMM and ML components.

        ML Architecture: {ml_architecture}

        Create:
        1. Step-by-step execution workflow
        2. Data flow between FEMM and ML components
        3. Computational requirements and runtime estimates
        4. Integration points and dependencies
        5. Validation and testing strategy
        6. Required software libraries and their purposes
        7. Potential failure points and mitigation strategies

        Ensure the plan is realistic and executable with available resources.
        """,
        expected_output="Complete integration plan with execution steps and resource requirements",
        agent=integration_architect,
        context=[ml_design_task],
    )

    # Create the crew
    crew = Crew(
        agents=[paper_analyzer, femm_geometry_planner, ml_architect, integration_architect],
        tasks=[paper_analysis_task, femm_planning_task, ml_design_task, integration_task],
        verbose=True,
        process=Process.sequential,
        planning=False,  # Disable planning for faster execution
    )

    return crew

def run_femm_planning(research_context_json: str) -> str:
    """Main function to run FEMM implementation planning"""
    try:
        # Parse research context
        research_context = json.loads(research_context_json)

        crew = create_femm_planning_crew()
        result = crew.kickoff(inputs={
            "research_context": json.dumps(research_context),
        })

        # Parse and validate the result
        try:
            result_text = result.raw

            # Try to extract JSON from the result
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())

                # Validate the plan structure
                required_fields = ['research_title', 'implementation_plan']
                if all(field in plan_data for field in required_fields):
                    return json.dumps(plan_data, indent=2)
                else:
                    return f"Invalid plan structure. Missing fields: {[f for f in required_fields if f not in plan_data]}"
            else:
                return result_text

        except json.JSONDecodeError:
            return result_text
        except Exception as e:
            return f"Error parsing plan result: {str(e)}"

    except Exception as e:
        return f"Error in FEMM planning: {str(e)}"

def create_sample_research_context() -> str:
    """Create sample research context for testing"""
    return json.dumps({
        "research_id": 3,
        "title": "CNN-based Topology Optimization for Electromagnetic Actuators",
        "authors": ["Smith, J.", "Lee, K.", "Chen, M."],
        "summary": "Proposes using convolutional neural networks for topology optimization of electromagnetic actuators, demonstrating significant improvements in force output and efficiency.",
        "methodology": "Uses FEMM for data generation, trains CNN on material distribution patterns, optimizes actuator topology using gradient-based methods",
        "femm_relevance": 9.2,
        "complexity": "Medium",
        "source_url": "https://arxiv.org/abs/2301.12345",
        "key_contributions": [
            "CNN architecture for topology optimization",
            "FEMM-based data generation pipeline",
            "15% improvement in actuator efficiency"
        ]
    })

if __name__ == "__main__":
    # Test the FEMM planning
    print("Testing FEMM implementation planning...")
    sample_context = create_sample_research_context()
    result = run_femm_planning(sample_context)
    print("Result:", result)
