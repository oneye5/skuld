package lazic.sources;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.WebHtmlGetter;

import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class NzBalanceOfPayments extends DataSourceBase {
	// The query includes "+Y" to fetch seasonally adjusted data as well
	private final String URL = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_BOP@DF_BOP,1.0/NZL.....Q.XDC.N+Y?startPeriod=2000-Q1&dimensionAtObservation=AllDimensions";

	@Override
	public Set<DataPoint> getDataPoints() {
		// 1. Fetch raw data
		String rawData = WebHtmlGetter.get(URL);

		// Optional: Dump to file for debugging/caching purposes
		try (FileWriter writer = new FileWriter("sample_data.txt")) {
			writer.write(rawData);
		} catch (IOException e) {
			System.err.println("Warning: Could not write sample_data.txt");
		}

		// 2. Parse JSON Tree
		JsonObject root = JsonParser.parseString(rawData).getAsJsonObject();
		JsonObject structure = root.getAsJsonObject("structure");
		JsonObject dataSet = root.getAsJsonArray("dataSets").get(0).getAsJsonObject();
		JsonObject observations = dataSet.getAsJsonObject("observations");

		// 3. Build Metadata Lookups (Dimensions & Attributes)

		// 3a. Dimensions (Used to decode the "0:0:0..." keys)
		JsonArray dimArray = structure.getAsJsonObject("dimensions").getAsJsonArray("observation");
		List<DimensionInfo> dimensions = new ArrayList<>();

		int timePeriodIndex = -1;
		int measureIndex = -1;
		int entryIndex = -1;
		int adjustmentIndex = -1;

		for (int i = 0; i < dimArray.size(); i++) {
			JsonObject dim = dimArray.get(i).getAsJsonObject();
			String id = dim.get("id").getAsString();

			// Map values for this dimension
			List<DimensionValue> values = new ArrayList<>();
			JsonArray valuesJson = dim.getAsJsonArray("values");
			for (JsonElement v : valuesJson) {
				JsonObject vObj = v.getAsJsonObject();
				String vId = vObj.get("id").getAsString();
				String vName = vObj.get("name").getAsString();
				String endStr = vObj.has("end") ? vObj.get("end").getAsString() : null;
				values.add(new DimensionValue(vId, vName, endStr));
			}
			dimensions.add(new DimensionInfo(id, values));

			// Identify key indices dynamically
			if ("TIME_PERIOD".equals(id)) timePeriodIndex = i;
			else if ("MEASURE".equals(id)) measureIndex = i;
			else if ("ACCOUNTING_ENTRY".equals(id)) entryIndex = i;
			else if ("ADJUSTMENT".equals(id)) adjustmentIndex = i;
		}

		// 3b. Attributes (Used to decode the [val, a, b, c] value arrays)
		// specifically looking for UNIT_MULT to scale values correctly
		JsonArray attrArray = structure.getAsJsonObject("attributes").getAsJsonArray("observation");
		int unitMultAttrIndex = -1;
		List<Double> unitMultipliers = new ArrayList<>();

		for (int i = 0; i < attrArray.size(); i++) {
			JsonObject attr = attrArray.get(i).getAsJsonObject();
			if ("UNIT_MULT".equals(attr.get("id").getAsString())) {
				unitMultAttrIndex = i;
				// Parse the multiplier definitions (e.g., ID "6" -> 10^6)
				for (JsonElement v : attr.getAsJsonArray("values")) {
					int exponent = Integer.parseInt(v.getAsJsonObject().get("id").getAsString());
					unitMultipliers.add(Math.pow(10, exponent));
				}
				break;
			}
		}

		// 4. Iterate Observations and build DataPoints
		Set<DataPoint> results = new HashSet<>();

		for (String key : observations.keySet()) {
			String[] indices = key.split(":");
			JsonArray obsValues = observations.getAsJsonArray(key);

			// -- Extract Value --
			JsonElement valEl = obsValues.get(0);
			if (valEl.isJsonNull()) continue;
			double value = valEl.getAsDouble();

			// -- Apply Unit Multiplier --
			// The observation array looks like: [value, status_idx, unit_mult_idx, currency_idx]
			// We need to map the unitMultAttrIndex (metadata) to the index in this array.
			// The attributes metadata list aligns with the observation array items starting from index 1.
			if (unitMultAttrIndex != -1 && (unitMultAttrIndex + 1) < obsValues.size()) {
				JsonElement multIdxEl = obsValues.get(unitMultAttrIndex + 1);
				if (!multIdxEl.isJsonNull()) {
					int multIdx = multIdxEl.getAsInt();
					if (multIdx >= 0 && multIdx < unitMultipliers.size()) {
						value = value * unitMultipliers.get(multIdx);
					}
				}
			}

			// -- Extract Date --
			if (timePeriodIndex == -1 || timePeriodIndex >= indices.length) continue;
			int timeValIdx = Integer.parseInt(indices[timePeriodIndex]);
			DimensionValue timeDim = dimensions.get(timePeriodIndex).values.get(timeValIdx);

			// Use the 'end' period for the timestamp (e.g., "2000-03-31T00:00:00")
			LocalDateTime date = LocalDateTime.parse(timeDim.end, DateTimeFormatter.ISO_DATE_TIME);

			// -- Construct Feature Name --
			// e.g., "Balance of Payments - Primary income - Net - Seasonally adjusted"
			StringBuilder featureName = new StringBuilder("BOP");

			if (measureIndex != -1) {
				featureName.append(" - ").append(dimensions.get(measureIndex).getValName(indices[measureIndex]));
			}
			if (entryIndex != -1) {
				featureName.append(" - ").append(dimensions.get(entryIndex).getValName(indices[entryIndex]));
			}
			if (adjustmentIndex != -1) {
				// Shorten common adjustments for readability
				String adj = dimensions.get(adjustmentIndex).getValName(indices[adjustmentIndex]);
				if (adj.toLowerCase().contains("seasonally adjusted")) {
					featureName.append(" (SA)");
				} else if (adj.toLowerCase().contains("neither")) {
					featureName.append(" (NSA)");
				} else {
					featureName.append(" (").append(adj).append(")");
				}
			}

			// Create DataPoint (Ticker is null for Macro data)
			results.add(new DataPoint(date, null, featureName.toString(), value));
		}

		return results;
	}

	// --- Helper classes for Metadata ---

	private static class DimensionInfo {
		String id;
		List<DimensionValue> values;

		DimensionInfo(String id, List<DimensionValue> values) {
			this.id = id;
			this.values = values;
		}

		String getValName(String indexStr) {
			int i = Integer.parseInt(indexStr);
			if (i >= 0 && i < values.size()) {
				return values.get(i).name;
			}
			return "Unknown";
		}
	}

	private static class DimensionValue {
		String id;
		String name;
		String end; // For time periods

		DimensionValue(String id, String name, String end) {
			this.id = id;
			this.name = name;
			this.end = end;
		}
	}
}