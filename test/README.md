## Testing
```bash
conda install pytest
mkdir ../s3
pytest
```

For developers and debugging:
```bash
mkdir ../s3
python3 test/create_data.py
python3 test/test_parameters.py
python3 test/test_interpolation.py

python3 test/test_netcdf2d1.py && python3 test/test_netcdf2d2.py && python3 test/test_netcdf2d3.py && python3 test/test_netcdf2d4.py
```
