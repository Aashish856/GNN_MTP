async function getVehicleInfo(registrationNumber, sessionToken = "idk") {
  const url = `https://www.carinfo.app/_next/data/il0c0JnsdllC44ronHnNF/rto-vehicle-registration-detail/rto-details/${registrationNumber}.json?rto=${registrationNumber}`;

  try {
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept': 'application/json',
        'Referer': 'https://www.carinfo.app/',
        'Cookie': `session-token=${sessionToken}`
      }
    });

    const text = await response.text();

    if (text.startsWith('<!DOCTYPE html>')) {
      console.error(`HTML page returned — likely invalid or expired session token`);
      return null;
    } else {
      const data = JSON.parse(text);
      const webSection = data?.pageProps?.rtoDetailsReponse?.webSections?.[0];

      if (webSection) {
        return {
          imageUrl: webSection.imageUrl,
          subtitle: webSection.message?.subtitle
        };
      } else {
        console.error('No webSections found in response');
        return null;
      }
    }
  } catch (err) {
    console.error(`Error fetching data for ${registrationNumber}:`, err.message);
    return null;
  }
}

// Test with async IIFE
(async () => {
  const res = await getVehicleInfo("RJ14CV0002");
  console.log('Vehicle info:', res);
})();
