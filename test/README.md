## Testing
```bash
conda install pytest
mkdir ../s3
pytest
```

For developers and debugging:
```bash
mkdir ../s3
python3 test/create_data_1.py
python3 test/create_data_1b.py
python3 test/create_data_2.py
python3 test/test_1_parameters.py
python3 test/test_2_interpolation.py
python3 test/test_3_data.py
python3 test/test_3b_data.py
python3 test/test_4_table.py
python3 test/test_5_export.py
python3 test/test_5b_export.py
python3 test/test_6_query.py

# python3 test/test_netcdf2d1.py && python3 test/test_netcdf2d2.py && python3 test/test_netcdf2d3.py && python3 test/test_netcdf2d4.py
```

