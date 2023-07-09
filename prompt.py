
RadQNLI_EN="Determine whether the given context sentence from a radiology report contains the answer to the provided question."
RadQNLI_ZH="判断给定的放射学报告中的上下文句子是否包含了提供的问题的答案。"

ImpressionGPT_EN="You are a chest radiologist that identifies the main findings and diagnosis or impression based on the given FINDINGS section of the chest X-ray report, which details the radiologists' assessment of the chest X-ray image. Please ensure that your response is concise and does not exceed the length of the FINDINGS. What are the main findings and diagnosis or impression based on the given Finding in chest X-ray report"
ImpressionGPT_ZH="你是一位胸部放射科医生，根据胸部X光检查报告中所提供的findings部分，识别主要的发现和诊断或评估，该部分详细说明了放射科医生对胸部X光图像的评估。请确保你的回答简明扼要，不超过“发现”部分的长度。基于胸部X光检查报告中给出的发现，主要的发现和诊断或印象是什么？"

DeID="""Task: Please anonymize the following clinical note.\n \
Specific Rules: Replace all the following information with the term "[redacted]": \
\nRedact any strings that might be a name or acronym or initials, patients' names, doctors' names, the names of the M.D. or Dr., \
\nRedact any pager names, medical staff names,
\nRedact any strings that might be a location or address, such as "3970 Longview Drive",
\nRedact any strings that look like "something years old" or "age 37",
\nRedact any dates and IDs and numbers and record dates, 
\nRedact clinic and hospital names,
\nRedact professions such as "manager",
\nRedact any contact information"""

prompt_en={
    "ImpressionGPT":ImpressionGPT_EN,
    "DeID":DeID,
    "RadQNLI":RadQNLI_EN,
}
prompt_zh={
    "ImpressionGPT":ImpressionGPT_ZH,
    "DeID":DeID,
    "RadQNLI":RadQNLI_ZH,
}