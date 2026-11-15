from hypothesis import settings

# "default" keeps CI fast; run `pytest --hypothesis-profile=stress` for a deeper search.
settings.register_profile("default", max_examples=40, deadline=None)
settings.register_profile("stress", max_examples=500, deadline=None)
settings.load_profile("default")
