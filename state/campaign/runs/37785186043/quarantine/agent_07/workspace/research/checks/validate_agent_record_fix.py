// Minimal validation fix for agent record creation
function ensureAgentRecordFields(agentRecord) {
  const requiredFields = [
    'question', 'action', 'changed', 'verified', 
    'unverified', 'observed_effect', 'uncertainty_targeted',
    'uncertainty_reduced', 'process_decision', 'research_result',
    'decision', 'next', 'candidate_tasks', 'complexity_added',
    'failure_class', 'task_selection_observation'
  ];
  
  const sanitizedRecord = { ...agentRecord };
  
  // Initialize empty arrays for list-type fields if missing
  requiredFields.forEach(field => {
    if (!(field in sanitizedRecord)) {
      if (field === 'changed' || field === 'verified' || field === 'unverified') {
        sanitizedRecord[field] = [];
      } else if (typeof requiredFields[field] === 'string') {
        sanitizedRecord[field] = '';
      } else {
        sanitizedRecord[field] = null;
      }
    }
  });
  
  return sanitizedRecord;
}