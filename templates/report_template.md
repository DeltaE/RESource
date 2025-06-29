# {{ title }}

{{ intro }}

{% for fig in figures %}
## {{ fig.title }}

<img src="{{ fig.image_data }}" alt="{{ fig.title }}" style="max-width:100%; height:auto;">

{{ fig.explanation }}

{% endfor %}
