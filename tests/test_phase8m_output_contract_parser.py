from agent.offline_runtime import _expected_output_files


def test_input_reference_inside_output_description_is_not_deliverable():
    instruction = """
# Example

## Input Files

- `swap_to_value.json`
- `conventions.json`

## Required Output Files

### `swap_valuation.json`

- `pv01` uses the bump specified in `swap_to_value.json`.

### `summary.json`

Write the summary here.
"""

    assert _expected_output_files(instruction) == {
        "swap_valuation.json",
        "summary.json",
    }
