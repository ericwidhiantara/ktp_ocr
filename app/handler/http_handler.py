from starlette.responses import JSONResponse

from app.models.schemas import BaseResp, Meta


class CustomHttpException(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message

        super().__init__(self.message)


def custom_exception(request, exc: CustomHttpException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=BaseResp(
            meta=Meta(code=exc.status_code, message=exc.message, error=True),
        ).model_dump()
    )
