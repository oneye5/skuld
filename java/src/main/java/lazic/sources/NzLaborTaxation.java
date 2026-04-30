package lazic.sources;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.gson.Gson;

import lazic.utils.ingest.Cadence;
import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.ReleaseDate;
import lazic.utils.ingest.ReleaseLag;
import lazic.utils.ingest.WebHtmlGetter;

public class NzLaborTaxation extends DataSourceBase {
	private final String URL = "https://sdmx.oecd.org/public/rest/data/OECD.CTP.TPS,DSD_TAX_WAGES_COU@DF_TW_COU,2.1/NZL...S_C0+S_C2.AW167+AW100+AW67._Z.A?startPeriod=2000&dimensionAtObservation=AllDimensions";

	// OECD Taxing Wages (annual): published ~12 months after reference year-end.
	private static final ReleaseLag RELEASE_LAG = ReleaseLag.months(12);

	/**
	 * Returns a set of DataPoint's. Ticker is null if the datapoint does not pertain to a particular ticker, such as macroeconomic data for example
	 * There are multiple DataPoint's in a time-series feature, and there may be multiple features returned overall.
	 */
	@Override
	public String getSourceName() { return "nz_labor_taxation"; }

	@Override
	public Set<DataPoint> getDataPoints() {
		Set<DataPoint> dataPoints = new HashSet<>();

		// 1. Fetch Raw Data
		String rawData = WebHtmlGetter.get(URL);

		if (rawData == null || rawData.isEmpty()) {
			System.err.println("NzLaborTaxation: No data retrieved from OECD source.");
			return dataPoints;
		}

		// 2. Parse JSON structure
		Gson gson = new Gson();
		SdmxResponse response;

		try {
			response = gson.fromJson(rawData, SdmxResponse.class);
		} catch (Exception e) {
			System.err.println("NzLaborTaxation: Failed to parse JSON: " + e.getMessage());
			return dataPoints;
		}

		// Safety checks to prevent NullPointerExceptions
		if (response == null || response.dataSets == null || response.dataSets.isEmpty()
						|| response.structure == null || response.structure.dimensions == null
						|| response.structure.dimensions.observation == null) {
			System.err.println("NzLaborTaxation: Invalid data structure received.");
			return dataPoints;
		}

		// 3. Extract Dimension lookups
		List<Dimension> dimensions = response.structure.dimensions.observation;

		// Based on the SDMX-JSON structure from OECD labor taxation data, the key positions are:
		// Index 0 = REF_AREA (Reference area, e.g., "NZL")
		// Index 1 = MEASURE (e.g., "Income tax", "Average tax wedge")
		// Index 2 = UNIT_MEASURE (e.g., "National currency", "Percentage of labour costs")
		// Index 3 = HOUSEHOLD_TYPE (e.g., "Single person, no children", "Single person, 2 children")
		// Index 4 = INCOME_PRINCIPAL (e.g., "100% of average wage", "167% of average wage")
		// Index 5 = INCOME_SPOUSE (e.g., "Not applicable")
		// Index 6 = FREQ (e.g., "Annual")
		// Index 7 = TIME_PERIOD (e.g., "2000", "2001", ...)
		if (dimensions.size() <= 7) {
			System.err.println("NzLaborTaxation: Dimensions list is too short to parse.");
			return dataPoints;
		}

		Dimension measureDim = dimensions.get(1);
		Dimension unitMeasureDim = dimensions.get(2);
		Dimension householdTypeDim = dimensions.get(3);
		Dimension incomePrincipalDim = dimensions.get(4);
		Dimension timeDim = dimensions.get(7);

		// 4. Iterate through observations and map to DataPoints
		Map<String, List<Double>> observations = response.dataSets.get(0).observations;

		if (observations == null) {
			System.err.println("NzLaborTaxation: No observations found.");
			return dataPoints;
		}

		for (Map.Entry<String, List<Double>> entry : observations.entrySet()) {
			String key = entry.getKey(); // e.g. "0:15:2:0:1:0:0:0"
			List<Double> values = entry.getValue();

			// The value array's 0 index holds the actual data value
			if (values == null || values.isEmpty() || values.get(0) == null) continue;
			Double value = values.get(0);

			// Parse key indices
			String[] indices = key.split(":");

			try {
				if (indices.length <= 7) continue;

				int measureIndex = Integer.parseInt(indices[1]);
				int unitMeasureIndex = Integer.parseInt(indices[2]);
				int householdTypeIndex = Integer.parseInt(indices[3]);
				int incomePrincipalIndex = Integer.parseInt(indices[4]);
				int timeIndex = Integer.parseInt(indices[7]);

				// Lookup relevant metadata
				String measureName = measureDim.values.get(measureIndex).name;
				String unitMeasure = unitMeasureDim.values.get(unitMeasureIndex).name;
				String householdType = householdTypeDim.values.get(householdTypeIndex).name;
				String incomePrincipal = incomePrincipalDim.values.get(incomePrincipalIndex).name;

				// Construct a comprehensive feature name combining all dimensions
				String featureName = String.format("NZ Labor Tax - %s (%s) - %s - %s",
								measureName, unitMeasure, householdType, incomePrincipal);

				// Get time information
				DimensionValue timeVal = timeDim.values.get(timeIndex);
				String timeStartStr = timeVal.start;

				// Parse Time (period start), then shift to release date.
				LocalDateTime periodStart;
				if (timeStartStr != null && !timeStartStr.isEmpty()) {
					periodStart = LocalDateTime.parse(timeStartStr, DateTimeFormatter.ISO_DATE_TIME);
				} else {
					// Fallback: if 'start' is missing, construct from 'id' (e.g., "2000")
					String timeId = timeVal.id;
					// For annual data, append Jan 1st
					periodStart = LocalDateTime.parse(timeId + "-01-01T00:00:00");
				}
				LocalDateTime timestamp = ReleaseDate.applyLag(periodStart, Cadence.ANNUAL, RELEASE_LAG);

				// Create DataPoint
				// Ticker is null as this is macroeconomic data
				dataPoints.add(new DataPoint(
								timestamp,
								null,
								featureName,
								value
				));

			} catch (IndexOutOfBoundsException | NumberFormatException | NullPointerException e) {
				// Log parsing errors but continue processing other points
				System.err.println("NzLaborTaxation: Skipping point " + key + ": " + e.getMessage());
			} catch (Exception e) {
				// Catch other unexpected exceptions
				System.err.println("NzLaborTaxation: Unexpected error for point " + key + ": " + e.getMessage());
			}
		}

		System.out.println("NzLaborTaxation: Successfully parsed " + dataPoints.size() + " data points.");
		return dataPoints;
	}

	// ==========================================
	// Internal POJOs for GSON Parsing
	// ==========================================

	private static class SdmxResponse {
		public List<DataSet> dataSets;
		public Structure structure;
	}

	private static class DataSet {
		// Maps keys like "0:15:2:0:1:0:0:0" to a list [Value, Status, etc.]
		public Map<String, List<Double>> observations;
	}

	private static class Structure {
		public Dimensions dimensions;
	}

	private static class Dimensions {
		// JSON path: structure -> dimensions -> observation
		public List<Dimension> observation;
	}

	private static class Dimension {
		public String id;
		public String name;
		public List<DimensionValue> values;
	}

	private static class DimensionValue {
		public String id;
		public String name;
		public String start;  // ISO datetime for time periods
		public String end;    // ISO datetime for time periods
	}
}
