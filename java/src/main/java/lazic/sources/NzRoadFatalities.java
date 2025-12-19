package lazic.sources;

import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.WebHtmlGetter;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.StringReader;
import java.time.YearMonth;
import java.time.format.DateTimeFormatter;
import java.util.HashSet;
import java.util.Set;

public class NzRoadFatalities extends DataSourceBase {

	private final String URL = "https://sdmx.oecd.org/public/rest/data/OECD.ITF,DSD_ST@DF_STFAT,1.0/NZL.M...ROAD...?dimensionAtObservation=AllDimensions";

	/**
	 * Returns a set of DataPoint's containing NZ Road Fatalities.
	 * Ticker is null as this is macroeconomic data.
	 */
	@Override
	public Set<DataPoint> getDataPoints() {
		Set<DataPoint> points = new HashSet<>();

		// 1. Fetch Raw Data
		String rawXml = WebHtmlGetter.get(URL);
		if (rawXml == null || rawXml.isEmpty()) {
			return points;
		}

		try {
			// 2. Prepare XML Parser
			DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
			// Note: We leave setNamespaceAware(false) by default to treat tags like "generic:Obs" as the literal tag name.
			DocumentBuilder builder = factory.newDocumentBuilder();
			Document doc = builder.parse(new InputSource(new StringReader(rawXml)));
			doc.getDocumentElement().normalize();

			// 3. Iterate through Observations (<generic:Obs>)
			NodeList observationNodes = doc.getElementsByTagName("generic:Obs");

			for (int i = 0; i < observationNodes.getLength(); i++) {
				Node obsNode = observationNodes.item(i);

				if (obsNode.getNodeType() == Node.ELEMENT_NODE) {
					Element obsElement = (Element) obsNode;

					// Extract Value
					String valueStr = null;
					NodeList valueNodes = obsElement.getElementsByTagName("generic:ObsValue");
					if (valueNodes.getLength() > 0) {
						Element valueElement = (Element) valueNodes.item(0);
						valueStr = valueElement.getAttribute("value");
					}

					// Extract Time Period from Key
					String timePeriodStr = null;
					NodeList keyNodes = obsElement.getElementsByTagName("generic:ObsKey");
					if (keyNodes.getLength() > 0) {
						Element keyElement = (Element) keyNodes.item(0);
						NodeList valueChildren = keyElement.getElementsByTagName("generic:Value");

						// Loop through generic:Value items to find TIME_PERIOD
						for (int k = 0; k < valueChildren.getLength(); k++) {
							Element val = (Element) valueChildren.item(k);
							if ("TIME_PERIOD".equals(val.getAttribute("id"))) {
								timePeriodStr = val.getAttribute("value");
								break;
							}
						}
					}

					// 4. Create DataPoint if we have valid data
					if (valueStr != null && timePeriodStr != null) {
						try {
							// Parse "YYYY-MM" (e.g., "2025-04")
							YearMonth ym = YearMonth.parse(timePeriodStr, DateTimeFormatter.ofPattern("yyyy-MM"));

							// Create DataPoint (Setting time to start of month)
							DataPoint point = new DataPoint(
											ym.atDay(1).atStartOfDay(),
											null, // Ticker is null for macro data
											"NZ Road Fatalities",
											Double.valueOf(valueStr)
							);
							points.add(point);

						} catch (Exception e) {
							System.err.println("Error parsing data point: " + timePeriodStr + " -> " + valueStr);
						}
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed to parse NZ Road Fatalities XML", e);
		}

		return points;
	}
}