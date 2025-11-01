import joblib
from pprint import pprint
p = joblib.load('models/hypertension_model.joblib')
print('Pipeline steps:', list(p.named_steps.keys()))
pre = p.named_steps.get('preprocessor')
print('Preprocessor type:', type(pre))
# Try to print transformer specs
if hasattr(pre, 'transformers'):
    print('\nColumnTransformer.transformers:')
    pprint(pre.transformers)
if hasattr(pre, 'named_transformers_'):
    print('\nColumnTransformer.named_transformers_:')
    pprint(pre.named_transformers_.keys())
# feature_names_in_ on pipeline or preprocessor
for obj_name, obj in [('pipeline', p), ('preprocessor', pre)]:
    if hasattr(obj, 'feature_names_in_'):
        print(f"{obj_name}.feature_names_in_:", getattr(obj, 'feature_names_in_'))
# Try to call get_feature_names_out if available
try:
    names = pre.get_feature_names_out()
    print('\npre.get_feature_names_out() -> length', len(names))
    print(names)
except Exception as e:
    print('\nget_feature_names_out not available:', e)

# Also print prototype X columns from training by checking pipeline steps
print('\nDone')
