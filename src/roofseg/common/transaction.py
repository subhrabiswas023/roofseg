from dataclasses import dataclass, field
from typing import Protocol, override


class Transaction(Protocol):
    @property
    def is_failed(self) -> bool: ...
    def stage(self) -> None: ...
    def commit(self) -> None: ...
    def recover_if_failed(self) -> None: ...


@dataclass
class BatchedTransaction(Transaction):
    transactions: list[Transaction] = field(default_factory=list)

    @override
    def stage(self) -> None:
        for tx in self.transactions:
            tx.stage()

    @override
    def commit(self) -> None:
        for tx in self.transactions:
            tx.commit()

    @property
    @override
    def is_failed(self) -> bool:
        return any(tx.is_failed for tx in self.transactions)

    @override
    def recover_if_failed(self) -> None:
        if not self.is_failed or not self.transactions:
            return

        first_transaction = self.transactions[0]

        if first_transaction.is_failed:
            for tx in self.transactions:
                tx.recover_if_failed()

        else:
            for tx in self.transactions:
                tx.commit()
