# {{ personal_info.name or "Your Name" }}

{% if personal_info.email or personal_info.phone or personal_info.address %}
**Contact Information:**
{% if personal_info.email %}📧 {{ personal_info.email }}{% endif %}
{% if personal_info.phone %}📞 {{ personal_info.phone }}{% endif %}
{% if personal_info.address %}📍 {{ personal_info.address }}{% endif %}
{% if personal_info.linkedin %}🔗 [LinkedIn]({{ personal_info.linkedin }}){% endif %}
{% if personal_info.github %}🔗 [GitHub]({{ personal_info.github }}){% endif %}

---
{% endif %}

{% if summary %}
## Professional Summary

{{ summary }}

---
{% endif %}

{% if experience %}
## Professional Experience

{% for exp in experience %}
### {{ exp.position or "Position Title" }}
**{{ exp.company or "Company Name" }}** {% if exp.location %}| {{ exp.location }}{% endif %}  
*{{ exp.start_date or "Start Date" }} - {{ exp.end_date or "End Date" }}*

{% if exp.responsibilities %}
{% for resp in exp.responsibilities %}
- {{ resp }}
{% endfor %}
{% endif %}

{% if exp.achievements %}
**Key Achievements:**
{% for achievement in exp.achievements %}
- {{ achievement }}
{% endfor %}
{% endif %}

{% if exp.technologies %}
**Technologies:** {{ exp.technologies | join(', ') }}
{% endif %}

{% endfor %}
---
{% endif %}

{% if education %}
## Education

{% for edu in education %}
### {{ edu.degree or "Degree" }}
**{{ edu.institution or "Institution" }}** {% if edu.location %}| {{ edu.location }}{% endif %}  
*{{ edu.start_date or "Start" }} - {{ edu.end_date or "End" }}*
{% if edu.gpa %}  
**GPA:** {{ edu.gpa }}{% endif %}

{% if edu.relevant_coursework %}
**Relevant Coursework:** {{ edu.relevant_coursework | join(', ') }}
{% endif %}

{% endfor %}
---
{% endif %}

{% if skills and (skills.technical or skills.soft or skills.tools) %}
## Skills

{% if skills.technical %}
**Technical Skills:** {{ skills.technical | join(', ') }}
{% endif %}

{% if skills.tools %}
**Tools & Technologies:** {{ skills.tools | join(', ') }}
{% endif %}

{% if skills.soft %}
**Soft Skills:** {{ skills.soft | join(', ') }}
{% endif %}

---
{% endif %}

{% if projects %}
## Projects

{% for project in projects %}
### {{ project.name or "Project Name" }}
{% if project.start_date or project.end_date %}*{{ project.start_date or "" }}{% if project.start_date and project.end_date %} - {% endif %}{{ project.end_date or "" }}*{% endif %}

{% if project.description %}
{{ project.description }}
{% endif %}

{% if project.technologies %}
**Technologies:** {{ project.technologies | join(', ') }}
{% endif %}

{% if project.url or project.github_url %}
{% if project.url %}[Live Demo]({{ project.url }}){% endif %}{% if project.url and project.github_url %} | {% endif %}{% if project.github_url %}[GitHub]({{ project.github_url }}){% endif %}
{% endif %}

{% endfor %}
---
{% endif %}

{% if certifications %}
## Certifications

{% for cert in certifications %}
- **{{ cert.name or "Certification Name" }}** - {{ cert.issuer or "Issuer" }} ({{ cert.date or "Date" }})
{% endfor %}

---
{% endif %}

{% if achievements %}
## Achievements

{% for achievement in achievements %}
- {{ achievement }}
{% endfor %}
{% endif %}