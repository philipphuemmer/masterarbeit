#!/usr/bin/env python3
"""
Convert all hardcoded Section~X.Y references to \ref{} links.
Run from repo root: python convert_section_refs.py
"""

import re
from pathlib import Path

def generate_label_from_title(title):
    """Convert subsection title to label format"""
    label = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    label = re.sub(r'\s+', '_', label)
    return f"subsec_{label.lower()}"

def get_all_subsections():
    """Extract all subsection titles and generate their labels"""
    main_tex = Path("thesis/main.tex").read_text()
    # Find all \subsection{...}\label{...}
    pattern = r'\\subsection\{([^}]+)\}\\label\{([^}]+)\}'
    sections = {}
    for match in re.finditer(pattern, main_tex):
        title = match.group(1)
        label = match.group(2)
        sections[title] = label
    return sections

def convert_section_references():
    """Convert Section~X.Y to Section~\ref{...}"""
    main_tex = Path("thesis/main.tex").read_text()

    # Mapping of Section numbers to expected labels (based on order in document)
    section_map = {
        # Chapter 2
        "2.1": "subsec_operational_setting",
        "2.2": "subsec_environment_and_disturbances",
        "2.3": "subsec_assumptions_and_cost_model",
        "2.4": "subsec_motivating_example",
        # Chapter 3
        "3.1": "subsec_maintenance_optimization_in_infrastructure_systems",
        "3.2": "subsec_stochastic_vehicle_routing_problems_and_dynamic_service_planning",
        "3.3": "subsec_approximate_dynamic_programming_for_dynamic_routing_problems",
        # Chapter 4
        "4.1": "subsec_markov_decision_process",
        "4.2": "subsec_stateactionvalue_structure",
        "4.3": "subsec_adp_foundations_and_the_offlineonline_approach",
        # Chapter 5
        "5.1": "subsec_shared_zone_selection_logic",
        "5.2": "subsec_myopic",
        "5.4": "subsec_cost_function_approximation",
        "5.5": "subsec_dynamic_balance",
        "5.6": "subsec_value_function_approximation",
        # Chapter 6
        "6.1": "subsec_simulation_design",
        "6.2": "subsec_routing_and_travel_times",
        "6.3": "subsec_data_basis",
        # Chapter 7
        "7.1": "subsec_policy_mechanism_comparison",
        "7.2": "subsec_impact_of_zone_selection_on_cost",
        "7.3": "subsec_vfa_rollout_performance_and_limitations",
        "7.4": "subsec_statistical_validation",
        # Chapter 8
        "8.1": "subsec_interpretation_of_results",
        "8.2": "subsec_limitations",
        "8.3": "subsec_practical_relevance_and_transferability",
        # Chapter 9
        "9.1": "subsec_summary_of_key_findings",
        "9.2": "subsec_contributions_of_this_thesis",
        "9.3": "subsec_fundamental_limitations",
        "9.4": "subsec_generalizability_and_transferability",
        "9.5": "subsec_open_research_questions_and_future_directions",
    }

    total_replaced = 0

    for section_num, label in section_map.items():
        # Match Section~X.Y or section~X.Y (with tilde)
        pattern1 = rf'(Section|section)~{re.escape(section_num)}\b'
        replacement1 = rf'Section~\ref{{{label}}}'
        count1 = len(re.findall(pattern1, main_tex))
        main_tex = re.sub(pattern1, replacement1, main_tex)

        # Also match Section X.Y (with space) if not already converted
        pattern2 = rf'(Section|section)\s{re.escape(section_num)}\b(?!~\\ref)'
        replacement2 = rf'Section~\ref{{{label}}}'
        count2 = len(re.findall(pattern2, main_tex))
        main_tex = re.sub(pattern2, replacement2, main_tex)

        total_replaced += count1 + count2
        if count1 + count2 > 0:
            print(f"  Section {section_num}: {count1 + count2} reference(s)")

    # Write back
    Path("thesis/main.tex").write_text(main_tex)
    return total_replaced

if __name__ == "__main__":
    print("Converting Section references to \\ref{} links...")
    total = convert_section_references()
    print(f"\nTotal references converted: {total}")
    print("Don't forget to recompile: cd thesis && latexmk -pdf main.tex")
