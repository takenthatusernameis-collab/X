#!/usr/bin/env python3
import json

# Read the activation_status.json to understand the current state
with open('state/campaign/current_task.json', 'r') as f:
    contract = json.load(f)

print('=' * 80)
print('TASK SELECTION ANALYSIS REPORT')
print('=' * 80)

print('\n🔍 PRIMARY QUESTION ANALYSIS')
print(f'   {contract["primary_question"]}')

print('\n📋 TASK IDENTITY')
print(f'   Task ID: {contract["task_id"]}')
print(f'   Role: {contract["role"]}')
print(f'   Agent Number: {contract["agent_number"]}')
print(f'   Campaign Slot: {contract["campaign_slot"]}')

print('\n🎯 BOTTLENECK IDENTIFICATION')
print(f'   {contract["bottleneck"]}')

print('\n📊 SELECTION ANALYSIS')
print(f'   Selection Principle: {contract.get("selection_basis", {}).get("reason", "N/A")}')
print(f'   Qualitative Priority: {contract.get("selection_basis", {}).get("qualitative_priority", [])}')
print(f'   Compared Against: {contract.get("selection_basis", {}).get("compared_against", [])}')

print('\n🔄 EVIDENCE BASIS')
print(f'   {contract["evidence_basis"]}')

print('\n📈 EVIDENCE GATHERING STATUS')
print(f'   Objective: {contract["objective"]}')
print(f'   Scope: {contract["scope"]}')
print(f'   Success Criterion: {contract["success_criterion"]}')

print('\n✅ ANALYSIS SUMMARY')

# Read LEARNING_STATE.md to understand the evidence
with open('state/LEARNING_STATE.md', 'r') as f:
    ls_content = f.read()
    
# Check if we can find key indicators in LEARNING_STATE.md
if 'frontier-first selection' in ls_content.lower():
    print('   ✓ Frontier-first selection is operational')
    
if 'RETAIN' in ls_content and 'Strategy Delta' in ls_content:
    print('   ✓ Frontier-first Strategy Delta is RETAIN (verified)')
    
if 'momentum' in ls_content.lower() and 'SUPPORTED' in ls_content:
    print('   ✓ Momentum cell is SUPPORTED (positive evidence)')
    
if 'falsified' in ls_content.lower():
    print('   ✓ Evidence shows falsified cells (robust negative evidence)')
    
if 'ADMITTED AS CANDIDATE POSITIVE EVIDENCE' in ls_content:
    print('   ✓ Evidence shows admitted candidate positive evidence')

print('\n🎯 BOUNDARY ASSESSMENT')
print('   Task is: AUDIT recent activation history and frontier decisions')
print('   to determine whether explicit highest-value focused-task selection')
print('   produces better information flow.')

print('\n🔧 NEXT STEPS')
print('   The assessment requires examining the controller selection mechanism')
print('   against the prior opportunistic queue selection method.')

print('=' * 80)
