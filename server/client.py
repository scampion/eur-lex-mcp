from zeep import Client as ZeepClient
from zeep.transports import AsyncTransport
from zeep.wsse.username import UsernameToken


class EurLexClient:
    """A client for the EUR-LEX SOAP Webservice."""

    WSDL_URL = "http://eur-lex.europa.eu/eurlex-ws/wsd/wsdl"

    def __init__(self, username, password):
        transport = AsyncTransport(None)
        self.client = ZeepClient(
            self.WSDL_URL, transport=transport, wsse=UsernameToken(username, password)
        )

    async def search(
        self, query: str, language: str = "en", page: int = 1, page_size: int = 10
    ) -> dict:
        """Performs an expert search on the EUR-LEX webservice."""
        try:
            response = await self.client.service.searchRequest(
                expertQuery=query, page=page, pageSize=page_size, searchLanguage=language
            )
            return self.client.helpers.serialize_object(response)
        except Exception as e:
            print(f"An error occurred while querying EUR-LEX: {e}", file=sys.stderr)
            return {"error": str(e)}
