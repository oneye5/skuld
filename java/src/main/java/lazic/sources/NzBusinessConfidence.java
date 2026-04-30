package lazic.sources;

import java.io.StringReader;
import java.time.YearMonth;
import java.time.format.DateTimeFormatter;
import java.util.HashSet;
import java.util.Set;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;

import lazic.utils.ingest.Cadence;
import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.ReleaseDate;
import lazic.utils.ingest.ReleaseLag;
import lazic.utils.ingest.WebHtmlGetter;

public class NzBusinessConfidence extends DataSourceBase {
	private final String URL = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CLI,4.1/NZL.M.......?dimensionAtObservation=AllDimensions&format=genericdata";

	// OECD CLI / business & consumer confidence (monthly): published ~5 days after month-end via OECD STES.
	// https://www.oecd.org/sdd/leading-indicators/
	private static final ReleaseLag RELEASE_LAG = ReleaseLag.of(5);

	/**
	 * Returns a set of DataPoint's. Ticker is null if the datapoint does not pertain to a particular ticker, such as macroeconomic data for example
	 * There are multiple DataPoint's in a time-series feature, and there may be multiple features returned overall.
	 */
	@Override
	public String getSourceName() { return "nz_business_confidence"; }

	@Override
	public Set<DataPoint> getDataPoints() {
		Set<DataPoint> result = new HashSet<>();
		String rawData = WebHtmlGetter.get(URL);

		if (rawData == null || rawData.isEmpty()) {
			return result;
		}

		try {
			// Initialize XML Parser
			DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
			// Namespace awareness helps if the parser struggles with "generic:Obs" vs "Obs"
			factory.setNamespaceAware(true);
			DocumentBuilder builder = factory.newDocumentBuilder();
			InputSource is = new InputSource(new StringReader(rawData));
			Document doc = builder.parse(is);

			doc.getDocumentElement().normalize();

			// The data is contained in <generic:Obs> tags
			// We use getElementsByTagNameNS to handle the "generic" namespace safely,
			// or strictly by name if namespace URI is not strictly required for simple extraction.
			NodeList obsList = doc.getElementsByTagName("generic:Obs");

			for (int i = 0; i < obsList.getLength(); i++) {
				Element obs = (Element) obsList.item(i);

				// 1. Extract Key values (Time and Measure)
				Element obsKey = (Element) obs.getElementsByTagName("generic:ObsKey").item(0);
				NodeList keyValues = obsKey.getElementsByTagName("generic:Value");

				String timePeriodStr = null;
				String measureCode = null;

				for (int k = 0; k < keyValues.getLength(); k++) {
					Element valElem = (Element) keyValues.item(k);
					String id = valElem.getAttribute("id");
					if ("TIME_PERIOD".equals(id)) {
						timePeriodStr = valElem.getAttribute("value");
					} else if ("MEASURE".equals(id)) {
						measureCode = valElem.getAttribute("value");
					}
				}

				// 2. Extract the numerical value
				Element obsValue = (Element) obs.getElementsByTagName("generic:ObsValue").item(0);
				String valStr = obsValue.getAttribute("value");

				// 3. Build the DataPoint
				if (timePeriodStr != null && measureCode != null && valStr != null) {
					// Parse Date: Format is YYYY-MM (e.g., 2001-02)
					// We convert this to the first day of that month at start of day
					YearMonth ym = YearMonth.parse(timePeriodStr, DateTimeFormatter.ofPattern("yyyy-MM"));

					// Determine Feature Name based on Measure Code
					// BCICP = Business Confidence, CCICP = Consumer Confidence
					String featureName = "OECD_" + measureCode;

					Double value = Double.valueOf(valStr);

					// Macro data usually has null ticker
					java.time.LocalDateTime periodStart = ym.atDay(1).atStartOfDay();
					java.time.LocalDateTime timestamp = ReleaseDate.applyLag(periodStart, Cadence.MONTHLY, RELEASE_LAG);
					DataPoint point = new DataPoint(
									timestamp,
									null,
									featureName,
									value
					);
					result.add(point);
				}
			}

		} catch (Exception e) {
			System.err.println("Error parsing XML for NzBusinessConfidence: " + e.getMessage());
			e.printStackTrace();
		}

		return result;
	}
}