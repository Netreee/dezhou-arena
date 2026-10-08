"""A narrow preflight for strict structured-output object requirements.

This checks closed objects and all-required properties without rewriting the
schema. It is not a compiler or a complete validator for every provider's JSON
Schema subset. Other provider-specific limits can still reject a request.
"""

from shared_logging import JsonObject, JsonValue

from poker.agent.react.backend import BackendError


def validate_strict_object_schemas(schema: JsonObject) -> None:
    """Reject open objects and optional properties anywhere in schema nodes.

    Nullable required values and explicit anyOf object variants preserve their
    declared meaning. An object with no properties may omit required, which is
    equivalent to an empty list. Annotation data (e.g. defaults or enum values)
    is not mistaken for a schema. Error messages never echo user schema values.
    """
    def visit(node: JsonValue) -> None:
        if not isinstance(node, dict):
            return
        kind = node.get("type")
        is_object = (kind == "object" or (isinstance(kind, list) and "object" in kind)
                     or "properties" in node or "additionalProperties" in node)
        if is_object:
            properties = node.get("properties", {})
            required = node.get("required", [])
            if node.get("additionalProperties") is not False:
                raise BackendError("Strict structured output requires closed object schemas")
            if (not isinstance(properties, dict) or not isinstance(required, list)
                    or any(not isinstance(value, str) for value in required)
                    or len(required) != len(properties) or set(required) != set(properties)):
                raise BackendError("Strict structured output requires every object property to be required")
        # These keyword values are mappings of names to child schemas; the
        # mapping itself is not a schema (a property can be named 'properties').
        for key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
            mapping = node.get(key)
            if isinstance(mapping, dict):
                for child in mapping.values():
                    visit(child)
        for key in ("items", "additionalProperties", "additionalItems", "contains", "propertyNames",
                    "not", "if", "then", "else", "unevaluatedProperties", "unevaluatedItems", "contentSchema"):
            child = node.get(key)
            if isinstance(child, list):
                for entry in child:
                    visit(entry)
            else:
                visit(child)
        for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
            children = node.get(key)
            if isinstance(children, list):
                for child in children:
                    visit(child)

    visit(schema)
