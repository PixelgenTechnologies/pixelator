{% if obj.display %}
   {% set doc_module = obj.id[:-(obj.qual_name|length + 1)] %}
   {% set defining_module = obj.obj["original_path"].rsplit(".", 1)[0] if obj.imported else doc_module %}
   {# Short names that AutoAPI may leave unqualified but are not local objects. #}
   {% set leave_unqualified = [
      'object', 'str', 'int', 'float', 'bool', 'bytes', 'list', 'dict', 'set',
      'tuple', 'type', 'Exception', 'BaseException', 'Generic', 'Protocol',
      'TypedDict', 'NamedTuple', 'Enum', 'IntEnum', 'Flag', 'IntFlag', 'ABC',
      'Iterable', 'Iterator', 'Mapping', 'Sequence', 'Callable', 'Any',
   ] %}
   {% if is_own_page %}
{{ obj.id }}
{{ "=" * obj.id | length }}

   {# Own-page class docs have no module context. AutoAPI shortens same-module
      bases assuming currentmodule is set; use qual_name (not the fully
      qualified id) with currentmodule so Sphinx does not double-prefix the
      registered object name. #}
.. py:currentmodule:: {{ doc_module }}

   {% endif %}
   {% set visible_children = obj.children|selectattr("display")|list %}
   {% set own_page_children = visible_children|selectattr("type", "in", own_page_types)|list %}
   {% if is_own_page and own_page_children %}
.. toctree::
   :hidden:

      {% for child in own_page_children %}
   {{ child.include_path }}
      {% endfor %}

   {% endif %}
.. py:{{ obj.type }}:: {% if is_own_page %}{{ obj.qual_name }}{% else %}{{ obj.short_name }}{% endif %}{% if obj.type_params %}[{{ obj.type_params }}]{% endif %}{% if obj.args %}({{ obj.args }}){% endif %}

   {% for (args, return_annotation) in obj.overloads %}
      {{ " " * (obj.type | length) }}   {{ obj.short_name }}{% if args %}({{ args }}){% endif %}

   {% endfor %}
   {% if obj.bases %}
      {% if "show-inheritance" in autoapi_options %}

   {# For imported-member pages, currentmodule is the import location, so
      re-qualify short local bases with the defining module. #}
   Bases: {% for base in obj.bases %}{% set base_root = base.split('[')[0] %}{{ (defining_module ~ '.' ~ base if obj.imported and '.' not in base_root and base_root not in leave_unqualified else base)|link_objs }}{% if not loop.last %}, {% endif %}{% endfor %}
      {% endif %}


      {% if "show-inheritance-diagram" in autoapi_options and obj.bases != ["object"] %}
   .. autoapi-inheritance-diagram:: {{ obj.obj["full_name"] }}
      :parts: 1
         {% if "private-members" in autoapi_options %}
      :private-bases:
         {% endif %}

      {% endif %}
   {% endif %}
   {% if obj.docstring %}

   {{ obj.docstring|indent(3) }}
   {% endif %}
   {% for obj_item in visible_children %}
      {% if obj_item.type not in own_page_types %}

   {{ obj_item.render()|indent(3) }}
      {% endif %}
   {% endfor %}
   {% if is_own_page and own_page_children %}
      {% set visible_attributes = own_page_children|selectattr("type", "equalto", "attribute")|list %}
      {% if visible_attributes %}
Attributes
----------

.. autoapisummary::

         {% for attribute in visible_attributes %}
   {{ attribute.id }}
         {% endfor %}


      {% endif %}
      {% set visible_exceptions = own_page_children|selectattr("type", "equalto", "exception")|list %}
      {% if visible_exceptions %}
Exceptions
----------

.. autoapisummary::

         {% for exception in visible_exceptions %}
   {{ exception.id }}
         {% endfor %}


      {% endif %}
      {% set visible_classes = own_page_children|selectattr("type", "equalto", "class")|list %}
      {% if visible_classes %}
Classes
-------

.. autoapisummary::

         {% for klass in visible_classes %}
   {{ klass.id }}
         {% endfor %}


      {% endif %}
      {% set visible_methods = own_page_children|selectattr("type", "equalto", "method")|list %}
      {% if visible_methods %}
Methods
-------

.. autoapisummary::

            {% for method in visible_methods %}
   {{ method.id }}
            {% endfor %}


      {% endif %}
   {% endif %}
{% endif %}
